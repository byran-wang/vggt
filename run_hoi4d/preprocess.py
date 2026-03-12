"""
Preprocess HOI4D sequences for BundleSDF.

Input  → HOI4D_ori/{ZY}/{H}/{C}/{N}/{S}/{s}/{T}/
Output → {output_dir}/
  train/{seq_name}/
    rgb/*.jpg         decoded RGB frames (copied)
    depth/*.png       16-bit uint16 depth in mm
    meta/*.pkl        per-frame: camMat(3×3), objTrans(3,)/None, objRot(3,)/None
    mask_hand
  masks_XMem/
    {seq_name}/*.png         binary 0/255 object masks
  models/{obj_key}/textured_simple.obj   copied from CAD source

This layout is HO3D_v3-compatible.  Ho3dReader / run_ho3d.py work through
the Hoi4dReader subclass in run_hoi4d/data_reader.py.

Usage:
  # Single sequence
  python run_hoi4d/preprocess.py --seq ZY20210800001/H1/C7/N1/S100/s01/T1

  # Multiple sequences from txt (one per line)
  python run_hoi4d/preprocess.py --seq_txt /path/to/seq_list.txt

  # Auto-discover all sequences
  python run_hoi4d/preprocess.py
"""
import os, sys, glob, json, pickle, shutil, argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rt

# HO3D depth encoding constant (same as Ho3dReader.get_depth)
_HO3D_DEPTH_SCALE = 0.00012498664727900177

# Convert OpenCV camera frame → OpenGL camera frame (matches HO3D stored convention)
_GLCAM_IN_CVCAM = np.array([[1, 0, 0, 0],
                             [0,-1, 0, 0],
                             [0, 0,-1, 0],
                             [0, 0, 0, 1]], dtype=np.float64)

# ── HOI4D-Instructions: pixel2category ───────────────────────────────────────
_INSTR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'HOI4D-Instructions', 'prepare_4Dseg')
sys.path.insert(0, _INSTR_DIR)
from pixel2category import f as _pixel2category   # core mask parser
# ─────────────────────────────────────────────────────────────────────────────

CAT_TO_CADNAME = {
    'C1': 'ToyCar', 'C2': 'Mug',    'C5': 'Bottle',
    'C7': 'Bowl',   'C12': 'Kettle', 'C13': 'Knife',
}


def seq_rel_to_name(seq_rel_path):
    """'ZY.../H1/C7/N1/S100/s01/T1' → 'ZY..._H1_C7_N1_S100_s01_T1'"""
    return seq_rel_path.strip('/').replace('/', '_')


def _list_files(dir_path, pattern):
    return sorted(glob.glob(os.path.join(dir_path, pattern)))


def _build_source_candidates(hoi4d_ori_dir, seq_rel_path, seq_name, kind):
    rel = seq_rel_path.strip('/')
    if kind == 'rgb':
        subdirs    = ('align_rgb', 'rgb', '')
        frame_glob = '*.jpg'
    else:
        subdirs    = ('align_depth', 'depth', '')
        frame_glob = '*.png'

    candidates = []
    for seq_key in (rel, seq_name):
        base = os.path.join(hoi4d_ori_dir, seq_key)
        for subdir in subdirs:
            candidates.append(os.path.join(base, subdir) if subdir else base)

    out  = []
    seen = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out, frame_glob


def _find_source_frame_dir(hoi4d_ori_dir, seq_rel_path, seq_name, kind):
    candidates, frame_glob = _build_source_candidates(
        hoi4d_ori_dir, seq_rel_path, seq_name, kind)
    for cand in candidates:
        if os.path.isdir(cand) and len(_list_files(cand, frame_glob)) > 0:
            return cand, frame_glob, candidates
    return '', frame_glob, candidates


def _ensure_frames_in_output(hoi4d_ori_dir, seq_rel_path, seq_name, out_dir, kind):
    if kind == 'rgb':
        dst_subdir, frame_glob, label, ext = 'rgb',   '*.jpg', 'RGB',   '.jpg'
    else:
        dst_subdir, frame_glob, label, ext = 'depth', '*.png', 'DEPTH', '.png'

    dst_dir = os.path.join(out_dir, dst_subdir)
    if os.path.isdir(dst_dir):
        n = len(_list_files(dst_dir, frame_glob))
        if n > 0:
            return n

    src_dir, frame_glob, candidates = _find_source_frame_dir(
        hoi4d_ori_dir, seq_rel_path, seq_name, kind)
    if not src_dir:
        print(f'    [WARN] {label} source not found. Tried: {"; ".join(candidates[:3])} ...')
        return 0

    os.makedirs(dst_dir, exist_ok=True)
    for src_f in sorted(_list_files(src_dir, frame_glob)):
        fid_raw = os.path.splitext(os.path.basename(src_f))[0]
        try:
            new_name = f'{int(fid_raw):04d}{ext}'
        except ValueError:
            new_name = os.path.basename(src_f)
        dst_f = os.path.join(dst_dir, new_name)
        if os.path.exists(dst_f):
            continue
        shutil.copy2(src_f, dst_f)
    return len(_list_files(dst_dir, frame_glob))


def _depth_uint16mm_to_ho3d(depth_mm):
    """Convert HOI4D uint16 mm depth → HO3D 3-channel packed BGR PNG.

    HO3D decoding (Ho3dReader.get_depth):
        depth_m = (bgr[...,2] + bgr[...,1]*256) * depth_scale

    Inverse:
        val = depth_mm / (1000 * depth_scale)
        R   = val % 256  → bgr[...,2]
        G   = val // 256 → bgr[...,1]
        B   = 0
    """
    val = np.round(depth_mm.astype(np.float64) / (1000.0 * _HO3D_DEPTH_SCALE)).astype(np.uint32)
    out = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    out[..., 2] = (val % 256).astype(np.uint8)    # R  (OpenCV BGR index 2)
    out[..., 1] = (val // 256).astype(np.uint8)   # G  (OpenCV BGR index 1)
    return out


def _convert_depth_frames_to_ho3d(hoi4d_ori_dir, seq_rel_path, seq_name, out_dir):
    """Read HOI4D uint16 depth PNGs, convert to HO3D format, write to out_dir/depth/."""
    dst_dir = os.path.join(out_dir, 'depth')
    if os.path.isdir(dst_dir) and len(_list_files(dst_dir, '*.png')) > 0:
        return len(_list_files(dst_dir, '*.png'))

    src_dir, _, candidates = _find_source_frame_dir(
        hoi4d_ori_dir, seq_rel_path, seq_name, kind='depth')
    if not src_dir:
        print(f'    [WARN] DEPTH source not found. Tried: {"; ".join(candidates[:3])} ...')
        return 0

    os.makedirs(dst_dir, exist_ok=True)
    for src_f in sorted(_list_files(src_dir, '*.png')):
        fid_raw = os.path.splitext(os.path.basename(src_f))[0]
        try:
            new_name = f'{int(fid_raw):04d}.png'
        except ValueError:
            new_name = os.path.basename(src_f)
        dst_f = os.path.join(dst_dir, new_name)
        if os.path.exists(dst_f):
            continue
        depth_mm = cv2.imread(src_f, cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            print(f'    [WARN] Cannot read depth: {src_f}')
            continue
        if depth_mm.ndim == 3:          # already colour? take first channel
            depth_mm = depth_mm[..., 0].astype(np.uint16)
        ho3d_depth = _depth_uint16mm_to_ho3d(depth_mm)
        cv2.imwrite(dst_f, ho3d_depth)

    return len(_list_files(dst_dir, '*.png'))

def parse_binary_mask(mask_path, cat):
    """RGB colour-coded HOI4D mask → (obj_mask, hand_mask) both uint8 0/255.

    obj_mask:  all regions with instance label != 2
    hand_mask: regions with instance label == 2  (hand)
    """
    img = cv2.imread(mask_path)
    if img is None:
        return None, None
    H, W = img.shape[:2]
    try:
        arrs, _, inst_labels = _pixel2category(mask_path, cat)
    except Exception:
        return np.zeros((H, W), dtype=np.uint8), np.zeros((H, W), dtype=np.uint8)

    obj_mask  = np.zeros((H, W), dtype=np.uint8)
    hand_mask = np.zeros((H, W), dtype=np.uint8)
    for arr, inst in zip(arrs, inst_labels):
        if inst == 2:
            hand_mask[arr] = 255
        else:
            obj_mask[arr]  = 255
    return obj_mask, hand_mask


# ── GT pose ──────────────────────────────────────────────────────────────────

def json_to_pose(json_path):
    """HOI4D objpose JSON → 4×4 pose in camera frame (metres).
    Returns None when isEffective == 0.
    """
    with open(json_path) as fh:
        data = json.load(fh)
    if data.get('isEffective', 0) == 0 or not data.get('dataList'):
        return None
    anno  = data['dataList'][0]
    t, r  = anno['center'], anno['rotation']
    trans = np.array([t['x'], t['y'], t['z']], dtype=np.float64)
    euler = np.array([r['x'], r['y'], r['z']], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rt.from_euler('XYZ', euler).as_matrix()
    T[:3, 3]  = trans
    return T


# ── CAD model ────────────────────────────────────────────────────────────────

def get_cad_model_path(cad_dir, cat, n_id):
    cat_name = CAT_TO_CADNAME.get(cat)
    if cat_name is None:
        return None
    obj = os.path.join(cad_dir, cat_name, f'{int(n_id[1:]):03d}.obj')
    return obj if os.path.exists(obj) else None


# ── Per-sequence processing ───────────────────────────────────────────────────

def _build_hand_pose_map(hoi4d_ori_dir, seq_rel_path):
    """Build {fid_4digit: {handPose, handBeta, handTrans}} from HOI4D right-hand pickles.

    HOI4D stores one pickle per annotated frame at:
      /mnt/hdd_volume/datasets/HOI4D/Hand_pose/handpose_right_hand/{ZY}/{H}/{C}/{N}/{S}/{s}/{T}/{frame_id}.pickle
    Not all frames are annotated; missing frames are simply absent from the map.
    """
    hand_dir = os.path.join(
        '/mnt/hdd_volume/datasets/HOI4D/Hand_pose/handpose_right_hand',
        seq_rel_path.strip('/'))
    hand_map = {}
    for pkl_f in glob.glob(os.path.join(hand_dir, '*.pickle')):
        fid_raw = os.path.basename(pkl_f).replace('.pickle', '')
        try:
            fid = f'{int(fid_raw):04d}'
        except ValueError:
            continue
        try:
            with open(pkl_f, 'rb') as fh:
                d = pickle.load(fh, encoding='latin1')
            hand_map[fid] = {
                'handPose':  np.array(d['poseCoeff'], dtype=np.float32),  # (48,) global(3)+pose(45)
                'handBeta':  np.array(d['beta'],      dtype=np.float32),  # (10,)
                'handTrans': np.array(d['trans'],     dtype=np.float32),  # (3,)
            }
        except Exception as e:
            print(f'    [WARN] hand pickle read error {pkl_f}: {e}')
    return hand_map


def preprocess_sequence(seq_rel_path, hoi4d_ori_dir, cad_dir, output_dir):
    parts          = seq_rel_path.strip('/').split('/')
    zy, cat, n_id  = parts[0], parts[2], parts[3]
    seq_name       = seq_rel_to_name(seq_rel_path)

    # ── HO3D-like output dirs ─────────────────────────────────────────────────
    train_dir     = os.path.join(output_dir, 'train', seq_name)
    meta_dst      = os.path.join(train_dir, 'meta')
    mask_dst      = os.path.join(output_dir, 'masks_XMem', seq_name)
    hand_mask_dst = os.path.join(train_dir, 'mask_hand')
    mask_obj_dst  = os.path.join(train_dir, 'mask_object')

    # Skip if already complete (rgb, meta counts match AND first meta has handPose)
    n_rgb_exist  = len(_list_files(os.path.join(train_dir, 'rgb'), '*.jpg'))
    n_meta_exist = len(_list_files(meta_dst, '*.pkl'))
    if n_rgb_exist > 0 and n_rgb_exist == n_meta_exist:
        # Check that hand data has already been merged
        _first_meta = sorted(_list_files(meta_dst, '*.pkl'))
        _has_hand = False
        if _first_meta:
            try:
                with open(_first_meta[0], 'rb') as _f:
                    _has_hand = 'handPose' in pickle.load(_f)
            except Exception:
                pass
        if _has_hand:
            print(f'  [SKIP] {seq_name}')
            return True

    print(f'  Processing {seq_name} ...')
    os.makedirs(train_dir,     exist_ok=True)
    os.makedirs(meta_dst,      exist_ok=True)
    os.makedirs(mask_dst,      exist_ok=True)
    os.makedirs(hand_mask_dst, exist_ok=True)
    os.makedirs(mask_obj_dst,  exist_ok=True)

    seq_ori_dir = os.path.join(hoi4d_ori_dir, seq_rel_path)

    # 1. RGB → train/{seq}/rgb/
    n_rgb = _ensure_frames_in_output(
        hoi4d_ori_dir, seq_rel_path, seq_name, train_dir, kind='rgb')
    # 1b. Depth → train/{seq}/depth/  (converted to HO3D 3-channel packed format)
    n_dep = _convert_depth_frames_to_ho3d(
        hoi4d_ori_dir, seq_rel_path, seq_name, train_dir)
    if n_rgb == 0:
        print(f'    [WARN] Missing pre-decoded RGB frames: {train_dir}/rgb/*.jpg')
        return False
    if n_dep == 0:
        print(f'    [WARN] Missing pre-decoded depth frames: {train_dir}/depth/*.png')
        return False

    # 2. Camera intrinsics (one intrin.npy per ZY device)
    K = np.load(os.path.join(hoi4d_ori_dir, zy, 'intrin.npy'))

    # 3. GT pose map: fid_str → T(4×4)
    # objpose JSONs use unpadded names (0.json, 1.json, ...) → normalise to 4-digit.
    pose_map = {}
    for pf in sorted(glob.glob(os.path.join(seq_ori_dir, 'objpose', '*.json'))):
        fid_raw = os.path.basename(pf).replace('.json', '')
        try:
            fid = f'{int(fid_raw):04d}'
        except ValueError:
            fid = fid_raw
        T = json_to_pose(pf)
        if T is not None:
            pose_map[fid] = T

    # 3b. Right-hand MANO map: fid_str → {handPose, handBeta, handTrans}
    hand_map = _build_hand_pose_map(hoi4d_ori_dir, seq_rel_path)
    if hand_map:
        print(f'    hand annotations: {len(hand_map)} frames')

    # 4. Meta pkl per frame  (camMat always present; obj pose & hand pose sparse)
    for rgb_file in _list_files(os.path.join(train_dir, 'rgb'), '*.jpg'):
        fid      = os.path.basename(rgb_file).replace('.jpg', '')
        pkl_path = os.path.join(meta_dst, f'{fid}.pkl')

        # Load existing meta or build from scratch
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                meta = pickle.load(f)
            if 'handPose' in meta:
                continue  # already complete
        else:
            T = pose_map.get(fid)
            if T is not None:
                T_gl = _GLCAM_IN_CVCAM @ T      # OpenCV → OpenGL (matches HO3D)
                rod, _ = cv2.Rodrigues(T_gl[:3, :3])
                meta = {'camMat': K, 'objTrans': T_gl[:3, 3], 'objRot': rod.flatten()}
            else:
                meta = {'camMat': K, 'objTrans': None, 'objRot': None}

        # Merge right-hand MANO params (None when frame has no annotation)
        hand = hand_map.get(fid)
        meta['handPose']  = hand['handPose']  if hand else None  # (48,) float32
        meta['handBeta']  = hand['handBeta']  if hand else None  # (10,) float32
        meta['handTrans'] = hand['handTrans'] if hand else None  # (3,)  float32

        with open(pkl_path, 'wb') as f:
            pickle.dump(meta, f)

    # 5. Binary masks → masks_XMem/{seq}/ and train/mask_hand/
    mask_src = os.path.join(seq_ori_dir, '2Dseg', 'shift_mask')
    for mf in sorted(glob.glob(f'{mask_src}/*.png')):
        fid_raw  = os.path.basename(mf).replace('.png', '')
        try:
            fid = f'{int(fid_raw):04d}'
        except ValueError:
            fid = fid_raw
        obj_out  = os.path.join(mask_dst,      f'{int(fid_raw):05d}.png')
        hand_out = os.path.join(hand_mask_dst, f'{fid}.png')
        if os.path.exists(obj_out) and os.path.exists(hand_out):
            continue
        obj_mask, hand_mask = parse_binary_mask(mf, cat)
        if obj_mask is not None:
            cv2.imwrite(obj_out,  obj_mask)
            cv2.imwrite(hand_out, hand_mask)
            # copy masks_XMem → mask_object/ with 4-digit naming
            link = os.path.join(mask_obj_dst, f'{fid}.png')
            if not os.path.exists(link):
                shutil.copy2(obj_out, link)

    # 6. CAD model
    cat_name  = CAT_TO_CADNAME.get(cat, cat)
    n_id_int  = int(n_id[1:])               # 'N50' → 50
    obj_key   = f'{cat_name}{n_id_int:03d}' # e.g. 'Bowl050'
    cad_src   = get_cad_model_path(cad_dir, cat, n_id)

    # models/{obj_key}/textured_simple.obj  (shared across sequences)
    model_dst = os.path.join(output_dir, 'models', obj_key, 'textured_simple.obj')
    os.makedirs(os.path.dirname(model_dst), exist_ok=True)
    if cad_src and not os.path.exists(model_dst):
        shutil.copy2(cad_src, model_dst)
    elif not cad_src:
        print(f'    [WARN] CAD model not found for {cat}/{n_id}')

    n_rgb  = len(_list_files(os.path.join(train_dir, 'rgb'),   '*.jpg'))
    n_dep  = len(_list_files(os.path.join(train_dir, 'depth'), '*.png'))
    n_meta = len(_list_files(meta_dst, '*.pkl'))
    n_mask = len(_list_files(mask_dst, '*.png'))
    print(f'    rgb={n_rgb}, depth={n_dep}, meta={n_meta}, masks={n_mask}')
    return True


def find_sequences(hoi4d_ori_dir, categories, devices):
    seqs = []
    for zy in devices:
        zy_dir = os.path.join(hoi4d_ori_dir, zy)
        if not os.path.isdir(zy_dir):
            continue
        for h in sorted(os.listdir(zy_dir)):
            if not h.startswith('H'):
                continue
            for cat in categories:
                cat_dir = os.path.join(zy_dir, h, cat)
                if not os.path.isdir(cat_dir):
                    continue
                for pose_dir in glob.glob(f'{cat_dir}/*/*/*/objpose'):
                    t_dir = os.path.dirname(pose_dir)
                    seqs.append(os.path.relpath(t_dir, hoi4d_ori_dir))
    return sorted(seqs)


def _read_list_file(list_path):
    entries = []
    with open(list_path, 'r', encoding='utf-8') as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '#' in line:
                line = line.split('#', 1)[0].strip()
            if line:
                entries.append((line_no, line))
    return entries


def _collect_target_seqs(args, categories, devices):
    seqs = []

    if args.seq:
        seqs.extend((0, s.strip()) for s in args.seq.split(',') if s.strip())

    if args.seq_txt:
        if not os.path.isfile(args.seq_txt):
            raise FileNotFoundError(f'--seq_txt file not found: {args.seq_txt}')
        seqs.extend(_read_list_file(args.seq_txt))

    if not seqs:
        return find_sequences(args.hoi4d_ori_dir, categories, devices)

    merged = []
    for line_no, seq in seqs:
        norm  = seq.strip('/').strip()
        parts = norm.split('/')
        if len(parts) != 7:
            src = '--seq' if line_no == 0 else f'{args.seq_txt}:{line_no}'
            print(f'[WARN] Skip invalid seq "{seq}" from {src} (expected 7 path parts)')
            continue
        merged.append(norm)

    return list(dict.fromkeys(merged))


def main():
    p = argparse.ArgumentParser(description='Preprocess HOI4D for BundleSDF (HO3D layout)')
    p.add_argument('--hoi4d_ori_dir', default='/mnt/hdd_volume/datasets/HOI4D_ori')
    p.add_argument('--cad_dir',       default='/mnt/hdd_volume/datasets/HOI4D_ori/HOI4D_CAD_Model_for_release/rigid')
    p.add_argument('--output_dir',    default='/mnt/hdd_volume/datasets/HOI4D_ori/HOI4D_processed')
    p.add_argument('--categories',    default='C1,C2,C5,C7,C12,C13',
                   help='C1=ToyCar C2=Mug C5=Bottle C7=Bowl C12=Kettle C13=Knife')
    p.add_argument('--devices',       default='ZY20210800001,ZY20210800002,ZY20210800003,ZY20210800004')
    p.add_argument('--seq',           default='',
                   help='Sequence relative path(s), comma-separated')
    p.add_argument('--seq_txt',       default='',
                   help='Path to txt file: one sequence relative path per line')
    p.add_argument('--out_list', default='', 
                   help='Path for output processed_sequences.txt. Default: {output_dir}/processed_sequences.txt')
    args = p.parse_args()

    categories = args.categories.split(',')
    devices    = args.devices.split(',')
    os.makedirs(args.output_dir, exist_ok=True)

    seqs = _collect_target_seqs(args, categories, devices)
    print(f'Found {len(seqs)} sequences for categories {categories}')

    ok = fail = 0
    for i, seq in enumerate(seqs):
        cat = seq.split('/')[2]
        print(f'\n[{i+1}/{len(seqs)}] {seq}  (cat={cat}={CAT_TO_CADNAME.get(cat, "")})')
        if preprocess_sequence(seq, args.hoi4d_ori_dir, args.cad_dir, args.output_dir):
            ok += 1
        else:
            fail += 1

    print(f'\nDone. OK={ok}, Failed={fail}')

    train_root = os.path.join(args.output_dir, 'train')
    list_path  = args.out_list if args.out_list else os.path.join(args.output_dir, 'processed_sequences.txt')
    processed  = (sorted(d for d in os.listdir(train_root)
                         if os.path.isdir(os.path.join(train_root, d)))
                  if os.path.isdir(train_root) else [])
    with open(list_path, 'w') as fh:
        fh.write('\n'.join(processed) + '\n')
    print(f'Sequence list → {list_path}')


if __name__ == '__main__':
    main()
