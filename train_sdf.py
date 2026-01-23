import argparse
import os
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import trimesh

from viewer.viewer_step import ObjDataProvider


class SDFNetwork(nn.Module):
    """MLP network for predicting signed distance values."""

    def __init__(self, input_dim=3, hidden_dim=256, num_layers=8, skip_layer=4):
        super().__init__()
        self.skip_layer = skip_layer

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == skip_layer:
                layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        input_x = x
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                x = torch.cat([x, input_x], dim=-1)
            x = F.relu(layer(x))
        sdf = self.output_layer(x)
        return sdf


class ColorNetwork(nn.Module):
    """MLP network for predicting RGB color."""

    def __init__(self, input_dim=3, hidden_dim=256, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.layers(x)
        rgb = torch.sigmoid(self.output_layer(x))
        return rgb


def get_rays(H, W, K, c2w):
    """Generate rays for each pixel in the image."""
    device = K.device
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )
    dirs = torch.stack(
        [(i - K[0, 2]) / K[0, 0], (j - K[1, 2]) / K[1, 1], torch.ones_like(i)], dim=-1
    )
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def sample_points_in_tsdf(rays_o, rays_d, near, far, n_samples, perturb=True):
    """Sample points along rays within TSDF bounds."""
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(rays_o.shape[0], n_samples)

    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals


def volume_render_sdf(sdf_net, color_net, pts, z_vals, rays_d, truncation=0.05):
    """Volume rendering with SDF-based density."""
    sdf = sdf_net(pts.reshape(-1, 3)).reshape(pts.shape[:-1])
    rgb = color_net(pts.reshape(-1, 3)).reshape(*pts.shape[:-1], 3)

    # Convert SDF to density using sigmoid
    density = torch.sigmoid(-sdf / truncation) / truncation

    # Compute alpha and transmittance
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e-3)], dim=-1)
    alpha = 1.0 - torch.exp(-density * dists)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]
    weights = alpha * transmittance

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)

    return rgb_map, depth_map, weights


def load_dataset(provider: ObjDataProvider):
    """Load dataset from ObjDataProvider."""
    # Get the last step data which contains final camera poses
    last_step = provider.steps[-1]
    data = last_step["data"]

    intrinsics = data.get("intrinsics")
    extrinsics = data.get("extrinsics")

    # Load images and masks
    images = []
    masks = []
    for img_path, mask_path in zip(provider.images, provider.masks):
        img = np.array(Image.open(img_path).convert("RGB")) / 255.0
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0
        images.append(img)
        masks.append(mask)

    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)

    # Convert extrinsics (w2c) to c2w
    c2w_list = []
    for ext in extrinsics:
        w2c = np.eye(4)
        w2c[:3] = ext[:3]
        c2w = np.linalg.inv(w2c)
        c2w_list.append(c2w)
    c2w_list = np.stack(c2w_list, axis=0)

    return {
        "images": torch.tensor(images, dtype=torch.float32),
        "masks": torch.tensor(masks, dtype=torch.float32),
        "intrinsics": torch.tensor(intrinsics, dtype=torch.float32),
        "c2w": torch.tensor(c2w_list, dtype=torch.float32),
    }


def compute_scene_bounds(c2w_list, scale=1.5):
    """Compute scene bounds from camera positions."""
    camera_positions = c2w_list[:, :3, 3]
    center = camera_positions.mean(dim=0)
    radius = (camera_positions - center).norm(dim=-1).max() * scale
    return center, radius


def visualize_dataset_in_rerun(images, intrinsics, c2w):
    """Visualize camera poses, intrinsics and images in rerun."""
    import rerun as rr
    rr.init("train_sdf", spawn=True)

    n_images, H, W, _ = images.shape
    for cam_idx in range(n_images):
        K = intrinsics[cam_idx].cpu().numpy()
        pose = c2w[cam_idx].cpu().numpy()
        img = (images[cam_idx].cpu().numpy() * 255).astype(np.uint8)

        rr.log(
            f"world/camera_{cam_idx}",
            rr.Pinhole(image_from_camera=K, resolution=[W, H]),
        )
        rr.log(
            f"world/camera_{cam_idx}",
            rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]),
        )
        rr.log(f"world/camera_{cam_idx}/image", rr.Image(img))


def visualize_rays_in_rerun(rays_o, rays_d, pts, far, scale, iter_idx, n_vis=50):
    """Visualize rays and sampled points in rerun."""
    import rerun as rr
    if iter_idx == 0:
        rr.init("train_sdf", spawn=True)

    n_vis = min(n_vis, len(rays_o))
    ray_origins = rays_o[:n_vis].cpu().numpy()
    ray_ends = (rays_o[:n_vis] + rays_d[:n_vis] * far * scale).cpu().numpy()
    rr.log(
        f"training/rays_{iter_idx}",
        rr.LineStrips3D(
            [np.stack([o, e]) for o, e in zip(ray_origins, ray_ends)],
            colors=[255, 255, 0],
        ),
    )
    # Visualize sampled points along rays
    pts_vis = pts[:n_vis].reshape(-1, 3).cpu().numpy()
    rr.log(
        f"training/sample_points_{iter_idx}",
        rr.Points3D(pts_vis, colors=[0, 255, 255], radii=0.005),
        static=True,
    )


def render_preview_image(
    sdf_net, color_net, H, W, K, pose, center, scale, near, far, n_samples, output_dir, iter_idx, device
):
    """Render a full preview image and save to disk."""
    preview_dir = output_dir / "preview"
    os.makedirs(preview_dir, exist_ok=True)

    with torch.no_grad():
        preview_img = torch.zeros(H, W, 3, device=device)
        preview_depth = torch.zeros(H, W, device=device)
        chunk_size = 4096
        all_pixels = torch.stack(
            torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1
        ).reshape(-1, 2).to(device)

        for chunk_start in range(0, len(all_pixels), chunk_size):
            chunk_pixels = all_pixels[chunk_start:chunk_start + chunk_size]
            chunk_rays_o, chunk_rays_d = get_rays(H, W, K, pose)
            chunk_rays_o = chunk_rays_o[chunk_pixels[:, 0], chunk_pixels[:, 1]]
            chunk_rays_d = chunk_rays_d[chunk_pixels[:, 0], chunk_pixels[:, 1]]
            chunk_rays_o = (chunk_rays_o - center) * scale
            chunk_rays_d = chunk_rays_d / chunk_rays_d.norm(dim=-1, keepdim=True)
            chunk_pts, chunk_z_vals = sample_points_in_tsdf(
                chunk_rays_o, chunk_rays_d, near * scale, far * scale, n_samples
            )
            chunk_rgb, chunk_depth, _ = volume_render_sdf(
                sdf_net, color_net, chunk_pts, chunk_z_vals, chunk_rays_d
            )
            preview_img[chunk_pixels[:, 0], chunk_pixels[:, 1]] = chunk_rgb
            preview_depth[chunk_pixels[:, 0], chunk_pixels[:, 1]] = chunk_depth

        preview_img_np = (preview_img.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(preview_img_np).save(preview_dir / f"preview_{iter_idx:05d}.png")


def train_sdf(
    sdf_net,
    color_net,
    dataset,
    output_dir,
    num_iters=5000,
    batch_size=1024,
    lr=1e-4,
    n_samples=64,
    device="cuda",
):
    """Train SDF and color networks."""
    sdf_net = sdf_net.to(device)
    color_net = color_net.to(device)

    optimizer = torch.optim.Adam(
        list(sdf_net.parameters()) + list(color_net.parameters()), lr=lr
    )

    images = dataset["images"].to(device)
    masks = dataset["masks"].to(device)
    intrinsics = dataset["intrinsics"].to(device)
    c2w = dataset["c2w"].to(device)
    
    if 0:
        visualize_dataset_in_rerun(images, intrinsics, c2w)

    n_images, H, W, _ = images.shape
    center, radius = compute_scene_bounds(c2w)
    near, far = 0.1, radius.item() * 2

    # Normalize scene to unit cube
    scale = 1.0 / radius.item()

    pbar = tqdm(range(num_iters), desc="Training SDF")
    for iter_idx in pbar:
        # Random image and pixel selection
        img_idx = torch.randint(0, n_images, (1,)).item()
        img = images[img_idx]
        mask = masks[img_idx]
        K = intrinsics[img_idx]
        pose = c2w[img_idx]

        # Sample pixels (prefer object region, ratio 10:1 object:background)
        n_obj_target = batch_size * 10 // 11
        n_bg_target = batch_size - n_obj_target

        obj_coords = torch.nonzero(mask > 0.5, as_tuple=False)
        if len(obj_coords) > n_obj_target:
            obj_sample_idx = torch.randperm(len(obj_coords))[:n_obj_target]
            obj_pixels = obj_coords[obj_sample_idx]
        else:
            obj_pixels = obj_coords

        # Random background pixels
        bg_coords = torch.nonzero(mask <= 0.5, as_tuple=False)
        if len(bg_coords) > n_bg_target:
            bg_sample_idx = torch.randperm(len(bg_coords))[:n_bg_target]
            bg_pixels = bg_coords[bg_sample_idx]
        else:
            bg_pixels = bg_coords

        if len(obj_pixels) > 0 and len(bg_pixels) > 0:
            pixels = torch.cat([obj_pixels, bg_pixels], dim=0)
        elif len(obj_pixels) > 0:
            pixels = obj_pixels
        else:
            pixels = bg_pixels

        # Get ray origins and directions
        rays_o, rays_d = get_rays(H, W, K, pose)
        rays_o = rays_o[pixels[:, 0], pixels[:, 1]]
        rays_d = rays_d[pixels[:, 0], pixels[:, 1]]

        # Normalize to scene bounds
        rays_o = (rays_o - center) * scale
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

        # Sample points along rays
        pts, z_vals = sample_points_in_tsdf(
            rays_o, rays_d, near * scale, far * scale, n_samples
        )

        # Visualize rays and points in rerun (every 10 iterations)
        if 0 and iter_idx % 10 == 0:
            visualize_rays_in_rerun(rays_o, rays_d, pts, far, scale, iter_idx)

        # Volume render
        rgb_pred, depth_pred, weights = volume_render_sdf(
            sdf_net, color_net, pts, z_vals, rays_d
        )

        # Ground truth
        rgb_gt = img[pixels[:, 0], pixels[:, 1]]
        mask_gt = mask[pixels[:, 0], pixels[:, 1]]

        # Losses
        rgb_loss = F.mse_loss(rgb_pred, rgb_gt)

        # Eikonal loss for SDF regularization
        pts_flat = pts.reshape(-1, 3).requires_grad_(True)
        sdf_vals = sdf_net(pts_flat)
        grad = torch.autograd.grad(
            outputs=sdf_vals,
            inputs=pts_flat,
            grad_outputs=torch.ones_like(sdf_vals),
            create_graph=True,
        )[0]
        eikonal_loss = ((grad.norm(dim=-1) - 1) ** 2).mean()

        # Free space loss (background should have positive SDF)
        bg_mask = mask_gt < 0.5
        if bg_mask.sum() > 0:
            bg_sdf = sdf_net(pts[bg_mask].reshape(-1, 3))
            free_space_loss = F.relu(-bg_sdf).mean()
        else:
            free_space_loss = torch.tensor(0.0, device=device)

        loss = rgb_loss + 0.1 * eikonal_loss + 0.01 * free_space_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_idx % 100 == 0:
            pbar.set_postfix(
                rgb=rgb_loss.item(), eik=eikonal_loss.item(), free=free_space_loss.item()
            )
            render_preview_image(
                sdf_net, color_net, H, W, K, pose, center, scale, near, far, n_samples, output_dir, iter_idx, device
            )

    return sdf_net, color_net, {"center": center.cpu(), "scale": scale}


def save_networks(sdf_net, color_net, scene_params, output_dir):
    """Save trained networks and scene parameters."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(sdf_net.state_dict(), os.path.join(output_dir, "sdf_network.pth"))
    torch.save(color_net.state_dict(), os.path.join(output_dir, "color_network.pth"))
    torch.save(scene_params, os.path.join(output_dir, "scene_params.pth"))
    print(f"Saved networks to {output_dir}")


def extract_mesh(sdf_net, scene_params, resolution=128, threshold=0.0, device="cuda"):
    """Extract mesh from SDF network using marching cubes."""
    import mcubes

    sdf_net.eval()
    center = scene_params["center"].to(device)
    scale = scene_params["scale"]

    # Create grid in normalized coordinates
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    z = torch.linspace(-1, 1, resolution)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    grid_pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).to(device)

    # Evaluate SDF in batches
    sdf_vals = []
    batch_size = 65536
    with torch.no_grad():
        for i in range(0, len(grid_pts), batch_size):
            batch = grid_pts[i : i + batch_size]
            sdf = sdf_net(batch)
            sdf_vals.append(sdf.cpu())
    sdf_vals = torch.cat(sdf_vals, dim=0).numpy().reshape(resolution, resolution, resolution)

    # Marching cubes
    vertices, triangles = mcubes.marching_cubes(sdf_vals, threshold)

    # Transform back to world coordinates
    vertices = vertices / resolution * 2 - 1  # [-1, 1]
    vertices = vertices / scale + center.cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    return mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Train SDF network from posed images")
    parser.add_argument(
        "--result_folder",
        type=str,
        required=True,
        help="Path to result folder with posed images",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/sdf_train", help="Output directory"
    )
    parser.add_argument("--num_iters", type=int, default=5000, help="Training iterations")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_samples", type=int, default=64, help="Number of samples per ray")
    parser.add_argument(
        "--mesh_resolution", type=int, default=128, help="Mesh extraction resolution"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


def main(args):
    # Load object dataset from ObjDataProvider
    provider = ObjDataProvider(Path(args.result_folder))
    dataset = load_dataset(provider)
    print(f"Loaded {len(dataset['images'])} images from {args.result_folder}")

    # Initialize networks
    sdf_net = SDFNetwork()
    color_net = ColorNetwork()

    # Train object color network and SDF network
    output_dir = Path(args.output_dir) / provider.get_seq_name()
    sdf_net, color_net, scene_params = train_sdf(
        sdf_net,
        color_net,
        dataset,
        output_dir,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        n_samples=args.n_samples,
        device=args.device,
    )

    # Save the trained object SDF and color network
    save_networks(sdf_net, color_net, scene_params, output_dir)

    # Extract object mesh from the trained SDF network and save
    mesh = extract_mesh(
        sdf_net, scene_params, resolution=args.mesh_resolution, device=args.device
    )
    mesh_path = output_dir / "extracted_mesh.obj"
    mesh.export(mesh_path)
    print(f"Saved extracted mesh to {mesh_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
