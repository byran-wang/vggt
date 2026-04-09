# TODO Plan

Use this file to track daily TODOs.

## How To Use

- Add one section per day.
- Keep tasks short and actionable.
- Mark tasks done with `[x]`.
- Add quick notes at end of day.

## SSH Connection

```sshconfig
Host 3090_server1
  HostName 10.30.47.2
  User shibo
```

---

## Daily Template

### YYYY-MM-DD

#### Top Priorities
- Priority 1
- Priority 2
- Priority 3

#### Tasks
- Task A
- Task B
- Task C

#### Follow-ups / Blockers
- Follow-up 1
- Blocker 1

#### Notes
- Wins:
- Issues:
- Plan for tomorrow:

---

## Current Week

### 2026-03-06

#### Top Priorities
- [x] rsync zed dataset /home/simba/Documents/dataset/ZED_wenxuan/ to 3090_server1://data1/shibo/Documents/dataset
- [x] rsync `third_party/FoundationStereo/pretrained_models/model_best_bp2.pth` to 3090_server1:/data1/shibo/Documents/project/vggt_wenxuan_new/third_party/FoundationStereo/pretrained_models
- [x] in robust_hoi_pipeline/pipeline_joint_opt.py/_rectify_pose(), optimize the pose with second-order gradient refinement
- [x] in third_party/bundlesdf/eval_vis_nvdiffrast.py, for each sequence read the mesh file textured_mesh.obj in output directory, camera pose from output/ob_in_cam, rgb image, intrinsic from the data directory then rendering a normal image by nvdiffrast_render and overlay the normal to the image as in third_party/FoundationPose/eval_vis_nvdiffrast.py. And add a vis_git to filter the invalid frames as in third_party/FoundationPose/eval_vis_nvdiffrast.py. third_party/FoundationPose/eval_vis_nvdiffrast.py, third_party/bundlesdf/eval_vis_nvdiffrast.py and robust_hoi_pipeline/pipeline_joint_opt_eval_vis_nvdiffrast.py should use the same code framework and abstract the same function to utils_simba/eval_vis.py. The only difference between these three files is the data reader.

#### Tasks
- [x] list model/checkpoint files under /home/simba/Documents/project/vggt/third_party
- [x] verify remote dataset after rsync is complete
- [x] handle rsync conflict for CUB1/depth (remote `depth -> depth_fs` symlink verified)
- [x] verify remote model checkpoint md5 matches local (`fed9cbbb6f139e64520153d1f469f698`)

#### Follow-ups / Blockers
- [x] rsync warning seen: could not make way for new symlink (CUB1/depth) (resolved)
- [x] no remaining blockers

#### Notes
- Wins: all planned rsync and verification items completed, including model checkpoint sync to `vggt_wenxuan_new`.
- Plan for tomorrow: run/validate the next pipeline stage on `3090_server1` with the synced assets.

### 2026-03-06

#### Top Priorities
- [x] for each seqence in /home/simba/Documents/project/vggt/third_party/bundlesdf/output, rsync the mesh_cleaned.obj and ob_in_cam to /home/simba/Documents/project/vggt/third_party/bundlesdf/output_ho3d
- [x] in robust_hoi_pipeline/eval_sum_vis.py, add the normal overlay video geneated by third_party/bundlesdf/eval_vis_nvdiffrast.py to the merged video
- [x] for each seqence in /home/simba/Documents/project/vggt/third_party/bundlesdf/output, rsync the textured_mesh.obj to /home/simba/Documents/project/vggt/third_party/bundlesdf/output_ho3d
- [x] in robust_hoi_pipeline/eval_sum_vis.py, add the method name to the merged video
- [x] in robust_hoi_pipeline/eval_sum_vis.py, add args.vis_method_name to determin whether to visulize the method_name. Visulize the method name by default.
- [x] rsync 3090_server1:/data1/shibo/Documents/project/hold/code/logs to third_party/hold/output_ho3d.
- [x] create a new file third_party/hold/eval_vis_nvdiffrast.py. In third_party/hold/eval_vis_nvdiffrast.py for each sequence read object mesh file and camera pose from output_ho3d, rgb image, intrinsic from the data directory then rendering a normal image by nvdiffrast_render and overlay the normal to the image as in third_party/FoundationPose/eval_vis_nvdiffrast.py. And add a vis_git to filter the invalid frames as in third_party/FoundationPose/eval_vis_nvdiffrast.py. third_party/hold/eval_vis_nvdiffrast.py shold call the commmon function from utils_simba/eval_vis.py.
- [x] In third_party/hold/code/eval_vis_nvdiffrast.py for each sequence read object mesh file and camera pose from `logs_ho3d`, rgb image and intrinsic from `data_ho3d`, render normal overlays with nvdiffrast via `utils_simba/eval_vis.py`, and resolve sequence-to-checkpoint mapping from `third_party/hold/docs/data_doc.md` so it can be called from `run_wonder_hoi.py` / `run_wonder_hoi.sh`.
- [x] In third_party/hold/code/eval_vis_nvdiffrast.py, the correpondences between render_index and frame_idx in the generated frame_map.json is defined in `data_ho3d/build/corres.txt`.
- [x] In third_party/hold/code/eval_vis_nvdiffrast.py, not only load the object mesh but also hand mesh to render the normal overlay image.
- [x] In third_party/hold/code/eval_vis_nvdiffrast.py, seal the hand for each frames and save the object mesh in object space in the first frame.
- [x] follow third_party/hold/docs/setup.md to run bash ./bash/download_data.sh to download hold preprossed data with export HOLD_USERNAME=swang457@connect.hkust-gz.edu.cn and export HOLD_PASSWORD=Simba67379325
- [x] in robust_hoi_pipeline/eval_sum_vis.py, add the normal overlay video geneated by third_party/hold/code/eval_vis_nvdiffrast.py to the merged video

### 2026-03-09

#### Top Priorities
- [x] in pipeline_neus_init.py, load the joint_opt_dir results according to the results generated by /home/simba/Documents/project/vggt/robust_hoi_pipeline/pipeline_joint_opt.py


### 2026-03-10
#### Top Priorities
- [x] rsync output_baseline to 3090_server1:/data1/shibo/Documents/project/vggt/
- [x] in robust_hoi_pipeline/pipeline_neus_init.py, add args.max_registered_frames. if args.max_registered_frames > -1, then only load the first add args.max_registered_frames proprocessed data for neus optimization.
- [x] in prepare_neus_data(), do not union object mask and hand mask to masks directory, but save the hand_mask to masks_hand directory
- [x] in third_party/instant-nsr-pl/datasets/robust_hoi.py, RobustHOIDatasetBase not only load the object mask but also hand mask which is generated by prepare_neus_data().
- [x] in preprocess_data() of third_party/instant-nsr-pl/systems/neus.py, only sample the rays in the object foreground mask and background mask, and exclude rays sampling rays in the hand mask.


### 2026-03-11
#### Top Priorities
- [x] in robust_hoi_pipeline/pipeline_joint_opt_vis.py, visulize the mesh generated by robust_hoi_pipeline/pipeline_neus_init.py, and the mesh path is in the `pipeline_neus_init/neus_training/joint_opt/save` directory; load the latest `.obj` and log it to `world/neus/mesh`
### 2026-03-10
#### Top Priorities
- [x] rsync `3090_server1:/data1/shibo/Documents/project/vggt/output[3_17_08_37][df071f4][SOTA][HashGrid_base_resolution_64][n_levels_14]/metrics_summary` to `output[3_17_08_37][df071f4][SOTA][HashGrid_base_resolution_64][n_levels_14]` by `rsync -as --ignore-existing --info=progress2`

### 2026-04-09
#### Top Priorities
- [x]compare output/metrics_summary/eval.txt and SOTA results in CLAUDE.md in the metrics of ADD AUC and SAM3D CD, and sort ADD AUC
