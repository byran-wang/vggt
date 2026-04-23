# Sequence config for xperience-10m dataset (DATASET=xper1m).
# scene_name format: `xp__{uuid}__{ep}`  (double-underscore separators, ep like "ep1")
# The list is populated at import time by scanning XPER1M_RAW_DIR
# (defaults to /mnt/afs/zirui/datasets/xperience-10m/data).

import os
import re


XPER1M_RAW_DIR = os.getenv(
    "XPER1M_RAW_DIR", "/mnt/afs/zirui/datasets/xperience-10m/data"
)

SCENE_PREFIX = "xp__"
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_EP_RE = re.compile(r"^ep\d+$")


def parse_scene_name(scene_name):
    """Parse `xp__{uuid}__{ep}` -> (uuid, ep). Returns (None, None) on mismatch."""
    if not scene_name.startswith(SCENE_PREFIX):
        return None, None
    body = scene_name[len(SCENE_PREFIX):]
    parts = body.split("__")
    if len(parts) != 2:
        return None, None
    uuid, ep = parts
    if not _UUID_RE.match(uuid) or not _EP_RE.match(ep):
        return None, None
    return uuid, ep


def make_scene_name(uuid, ep):
    return f"{SCENE_PREFIX}{uuid}__{ep}"


def raw_episode_dir(scene_name, raw_root=None):
    """Map scene_name -> raw episode path under XPER1M_RAW_DIR."""
    raw_root = raw_root or XPER1M_RAW_DIR
    uuid, ep = parse_scene_name(scene_name)
    if uuid is None:
        raise ValueError(f"bad xper1m scene_name: {scene_name}")
    return os.path.join(raw_root, uuid, ep)


def _scan_sequences(raw_root):
    """Enumerate all (uuid, ep) pairs under raw_root. Ordered for stability."""
    if not os.path.isdir(raw_root):
        return []
    names = []
    for uuid in sorted(os.listdir(raw_root)):
        uuid_dir = os.path.join(raw_root, uuid)
        if not (os.path.isdir(uuid_dir) and _UUID_RE.match(uuid)):
            continue
        for ep in sorted(os.listdir(uuid_dir)):
            if _EP_RE.match(ep) and os.path.isdir(os.path.join(uuid_dir, ep)):
                names.append(make_scene_name(uuid, ep))
    return names


def _read_whitelist(path):
    """Read a whitelist file (one scene_name per line; blanks / # comments ignored).
    Returns the ordered list of valid scene names, or None if the file does not exist."""
    if not path or not os.path.isfile(path):
        return None
    names = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if parse_scene_name(s) != (None, None):
                names.append(s)
    return names


# sequence_name_list source precedence:
#   1. XPER1M_WHITELIST env var -> file with one scene per line (the common case,
#      built by generator/scripts/xper1m_build_whitelist.py and filtered to eps
#      that have enough right-wrist-visible frames)
#   2. Fallback: os.listdir scan of XPER1M_RAW_DIR (all 1457 eps, no visibility filter)
XPER1M_WHITELIST = os.getenv("XPER1M_WHITELIST", "")
_wl = _read_whitelist(XPER1M_WHITELIST) if XPER1M_WHITELIST else None
if _wl is not None:
    sequence_name_list = _wl
    print(f"[sequence_config_xper1m] whitelist {XPER1M_WHITELIST}: {len(sequence_name_list)} scenes")
else:
    sequence_name_list = _scan_sequences(XPER1M_RAW_DIR)

_DEFAULT = {
    "geometry_poor_frames": [],
    "cond_idx": 0,
    "cond_select_strategy": "manual",
    "frame_star": 0,
    "frame_end": 9999,
    "frame_interval": int(os.getenv("XPER1M_FRAME_INTERVAL", "1")),
    "frame_number": int(os.getenv("XPER1M_FRAME_NUMBER", "50")),
    "obj_num": 1,
    "obj_1_cond_idx": 0,
    "obj_2_cond_idx": 0,
    "obj_3_cond_idx": 0,
    "obj_4_cond_idx": 0,
}

sequences = {"default": dict(_DEFAULT)}
for name in sequence_name_list:
    sequences[name] = dict(_DEFAULT)
