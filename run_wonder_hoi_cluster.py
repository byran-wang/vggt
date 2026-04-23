"""Cluster-friendly pipeline driver for rhoi (hand-object interaction).

This is a refactored subset of run_wonder_hoi.py:
  - Only the stages that were still exercised by run_wonder_hoi.sh are kept.
  - Exploratory dead paths (hunyuan / hunyuan_omni / bundlesdf py38 / mpsfm /
    threestudio / cutie / realsense / hot3d / sync-to-local / inpaint / 100DoH)
    are removed.
  - Every subprocess call goes through `CondaEnv.run(...)` instead of
    hand-rolled `f"cd X && $(envs/Y/bin/python) ..."` + bare `os.system`.

Env layout is parameterised so the same code runs in two deployment shapes:

  A) Conda (dev boxes)         envs live at `$ENVS_DIR/<name>` where ENVS_DIR
                               defaults to `~/miniconda3/envs`.
  B) Docker / flat venv tree   ht-dm bakes envs under `/root/envs/<name>`.
                               Caller just sets `ENVS_DIR=/root/envs`.

Logical env role -> env-var override:
  main  (default `rhoi`, was `vggsfm_tmp` locally) -> $RHOI_ENV
  fs    (default `foundation_stereo`)              -> $FS_ENV
  fp    (default `foundation_pose`)                -> $FP_ENV
  sam3  (default `sam3`)                           -> $SAM3_ENV
  sam3d (default `sam3d-objects`)                  -> $SAM3D_ENV
  hamer (default `hamer`)                          -> $HAMER_ENV
"""

import argparse
import json
import os
import shlex
from types import SimpleNamespace

from confs.sequence_config import (
    dataset_dir,
    dataset_type,
    sequence_name_list,
    sequences,
    vggt_code_dir,
)


# ---------------------------------------------------------------------------
# Env abstraction
# ---------------------------------------------------------------------------

ENVS_DIR = os.getenv("ENVS_DIR", os.path.expanduser("~/miniconda3/envs"))


class CondaEnv:
    """A conda env (or a conda-packed env living under `/root/envs/`).

    Callers only need `.python` (absolute path to the env's python) and
    `.run(args, cwd=..., env_vars=..., shell_prefix=...)` which wraps
    `os.system` with stdout mirroring and rc propagation (SystemExit(1) on
    non-zero, so shell scripts with `|| exit 1` actually trip).
    """

    def __init__(self, name, envs_dir=None):
        self.name = name
        self.envs_dir = envs_dir or ENVS_DIR
        self.root = f"{self.envs_dir}/{self.name}"
        self.python = f"{self.root}/bin/python"

    def run(self, args, cwd=None, env_vars=None, shell_prefix=None, check=True):
        """Run `python args...` in this env. `args` must be a list[str]; values
        are shell-quoted so paths with spaces are safe. Returns the raw
        os.system return code for the benefit of callers that want to branch."""
        if not isinstance(args, (list, tuple)):
            raise TypeError(f"args must be list/tuple, got {type(args).__name__}")
        cmd = " ".join([shlex.quote(self.python)] + [shlex.quote(str(a)) for a in args])
        if env_vars:
            cmd = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_vars.items()) + " " + cmd
        if shell_prefix:
            cmd = f"{shell_prefix} && {cmd}"
        if cwd:
            cmd = f"cd {shlex.quote(cwd)} && {cmd}"
        print(cmd)
        rc = os.system(cmd)
        if check and rc != 0:
            raise SystemExit(1)
        return rc


def _make_envs():
    return SimpleNamespace(
        main=CondaEnv(os.getenv("RHOI_ENV", "rhoi")),
        fs=CondaEnv(os.getenv("FS_ENV", "foundation_stereo")),
        fp=CondaEnv(os.getenv("FP_ENV", "foundation_pose")),
        sam3=CondaEnv(os.getenv("SAM3_ENV", "sam3")),
        sam3d=CondaEnv(os.getenv("SAM3D_ENV", "sam3d-objects")),
        hamer=CondaEnv(os.getenv("HAMER_ENV", "hamer")),
    )


def _shell(cmd, check=True):
    """Run a non-python shell command (rm, ln, cd, rsync, ...). Consistent
    print+rc+SystemExit semantics as CondaEnv.run."""
    print(cmd)
    rc = os.system(cmd)
    if check and rc != 0:
        raise SystemExit(1)
    return rc


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


class run_wonder_hoi:
    def __init__(self, args, extras):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.code_dir = os.path.join(self.current_dir, "..")
        self.seq_list = args.seq_list
        self.execute_list = args.execute_list
        self.process_list = args.process_list
        self.dataset_dir = dataset_dir
        self.reconstruction_dir = args.reconstruction_dir
        self.rebuild = args.rebuild
        self.vis = args.vis
        self.eval = args.eval
        self.extras = extras
        self.envs = _make_envs()

        self.process_mapping = {
            "data_read": {
                "ZED_read_data": self.ZED_read_data,
            },
            "data_convert": {
                "ZED_parse_data": self.ZED_parse_data,
                "convert_depth_to_ply": self.convert_depth_to_ply,
                "soft_link_depth": self.soft_link_depth,
                "get_depth_from_foundation_stereo": self.get_depth_from_foundation_stereo,
                "ho3d_get_hand_mask_prompt": self.ho3d_get_hand_mask_prompt,
                "ho3d_get_obj_mask_prompt": self.ho3d_get_obj_mask_prompt,
                "ho3d_get_hand_mask": self.ho3d_get_hand_mask,
                "ho3d_get_obj_mask": self.ho3d_get_obj_mask,
                "ho3d_estimate_hand_pose": self.ho3d_estimate_hand_pose,
                "ho3d_interpolate_hamer": self.ho3d_interpolate_hamer,
                "xper1m_convert": self.xper1m_convert,
            },
            "obj_process": {
                "ho3d_obj_SAM3D_filter_2D": self.ho3d_obj_SAM3D_filter_2D,
                "ho3d_obj_SAM3D_filter_3D": self.ho3d_obj_SAM3D_filter_3D,
                "ho3d_obj_SAM3D_gen": self.ho3d_obj_SAM3D_gen,
                "ho3d_align_SAM3D_mask": self.ho3d_align_SAM3D_mask,
                "ho3d_align_SAM3D_pts": self.ho3d_align_SAM3D_pts,
                "ho3d_align_SAM3D_fp": self.ho3d_align_SAM3D_fp,
                "ho3d_align_by_foundation_pose": self.ho3d_align_by_foundation_pose,
                "pipeline_sam3d_align_filter": self.pipeline_sam3d_align_filter,
                "pipeline_sam3d_delete_unused": self.pipeline_sam3d_delete_unused,
                "pipeline_sam3d_best_id": self.pipeline_sam3d_best_id,
                "pipeline_sam3d_best_id_sum": self.pipeline_sam3d_best_id_sum,
                "ho3d_SAM3D_post_process": self.ho3d_SAM3D_post_process,
                "ho3d_align_gen_3d": self.ho3d_align_gen_3d,
                "ho3d_align_gen_3d_omni": self.ho3d_align_gen_3d_omni,
                "ho3d_obj_sdf_optimization": self.ho3d_obj_sdf_optimization,
                "ho3d_eval_intrinsic": self.ho3d_eval_intrinsic,
                "ho3d_eval_trans": self.ho3d_eval_trans,
                "ho3d_eval_rot": self.ho3d_eval_rot,
                "hoi_pipeline_data_preprocess": self.hoi_pipeline_data_preprocess,
                "hoi_pipeline_data_preprocess_sam3d_neus": self.hoi_pipeline_data_preprocess_sam3d_neus,
                "hoi_pipeline_get_corres": self.hoi_pipeline_get_corres,
                "hoi_pipeline_eval_corres": self.hoi_pipeline_eval_corres,
                "hoi_pipeline_neus_init": self.hoi_pipeline_neus_init,
                "hoi_pipeline_neus_global": self.hoi_pipeline_neus_global,
                "hoi_pipeline_joint_opt": self.hoi_pipeline_joint_opt,
                "hoi_pipeline_joint_opt_global": self.hoi_pipeline_joint_opt_global,
                "hoi_pipeline_reg_remaining": self.hoi_pipeline_reg_remaining,
                "hoi_pipeline_align_hand_object_h": self.hoi_pipeline_align_hand_object_h,
                "hoi_pipeline_align_hand_object_r": self.hoi_pipeline_align_hand_object_r,
                "hoi_pipeline_align_hand_object_o": self.hoi_pipeline_align_hand_object_o,
                "hoi_pipeline_align_hand_object_ho": self.hoi_pipeline_align_hand_object_ho,
                "hoi_pipeline_eval": self.hoi_pipeline_eval,
                "hoi_pipeline_eval_vis": self.hoi_pipeline_eval_vis,
                "hoi_pipeline_eval_vis_gt": self.hoi_pipeline_eval_vis_gt,
                "hoi_pipeline_teaser": self.hoi_pipeline_teaser,
                "eval_sum_intrinsic": self.eval_sum_intrinsic,
                "eval_sum_trans": self.eval_sum_trans,
                "eval_sum_rot": self.eval_sum_rot,
                "eval_sum": self.eval_sum,
                "eval_sum_vis": self.eval_sum_vis,
            },
            "hand_pose_postprocess": {
                "fit_hand_intrinsic": self.fit_hand_intrinsic,
                "fit_hand_trans": self.fit_hand_trans,
                "fit_hand_intrinsic_vis": self.fit_hand_intrinsic_vis,
                "fit_hand_trans_vis": self.fit_hand_trans_vis,
            },
            "baseline": {
                "foundation_pose_eval_vis": self.foundation_pose_eval_vis,
                "bundle_sdf_eval_vis": self.bundle_sdf_eval_vis,
                "hold_eval_vis": self.hold_eval_vis,
                "gt_eval_vis": self.gt_eval_vis,
            },
        }

    def run(self):
        """Stage-outer / seq-inner loop with per-seq isolation.
        A failure on one seq is caught, logged, and excludes that seq from all
        remaining process steps in this invocation — but other seqs continue.
        Python always exits 0 unless the user sends SIGINT; the shell driver
        relies on per-seq log inspection (Traceback / completion markers) for
        final OK/FAIL verdict, not on this script's exit code."""
        import traceback as _tb
        failed = {}  # seq -> "exe:process: <short error>"
        for exe in self.execute_list:
            for process in self.process_list:
                for seq in self.seq_list:
                    if seq in failed:
                        continue
                    self.seq_config = dict(sequences["default"])
                    if seq in sequences:
                        for k, v in sequences[seq].items():
                            self.seq_config[k] = v
                    try:
                        self.process_mapping[exe][process](seq, **self.extras)
                    except KeyboardInterrupt:
                        # User interrupt: propagate immediately, don't swallow.
                        print("\n[run] KeyboardInterrupt — aborting.", flush=True)
                        raise
                    except SystemExit as e:
                        # Subprocess failure surfaced by CondaEnv.run(). Log and
                        # skip this seq for the rest of this Python invocation.
                        err = f"SystemExit({e.code})"
                        print(
                            f"\n[run] FAIL {exe}:{process}({seq}): {err}\n"
                            f"[run] skipping {seq} for remaining stages in this invocation.",
                            flush=True,
                        )
                        failed[seq] = f"{exe}:{process}: {err}"
                    except Exception as e:
                        _tb.print_exc()
                        err = f"{type(e).__name__}: {e}"
                        print(
                            f"\n[run] FAIL {exe}:{process}({seq}): {err}\n"
                            f"[run] skipping {seq} for remaining stages in this invocation.",
                            flush=True,
                        )
                        failed[seq] = f"{exe}:{process}: {err}"
        if failed:
            print(f"\n[run] SUMMARY: {len(failed)} seq(s) failed in this invocation:", flush=True)
            for seq, where in sorted(failed.items()):
                print(f"  - {seq}: {where}", flush=True)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def print_header(self, process):
        header = f"========== start: {process} =========="
        bar = "-" * len(header)
        # flush=True is load-bearing: the shell driver's split_per_seq pipe
        # routes output to per-seq logs based on these headers. If they're
        # buffered, they arrive AFTER the subprocess (sam3, FS, HAMER, ...)
        # already streamed, and the splitter never sees them in time → every
        # line falls through to the worker fallback log.
        print(bar, flush=True)
        print(header, flush=True)
        print(bar, flush=True)

    def _get_best_cond_id(self, scene_name) -> str:
        """4-digit condition-frame id for this seq. Mode:
          manual  -> seq_config['cond_idx']
          auto    -> first line of SAM3D_align_filter/best_id.txt
        """
        strategy = self.seq_config["cond_select_strategy"]
        if strategy == "manual":
            return f"{self.seq_config['cond_idx']:04d}"
        if strategy == "auto":
            best = f"{self.dataset_dir}/{scene_name}/SAM3D_align_filter/best_id.txt"
            with open(best, "r") as f:
                return f.read().strip()
        raise ValueError(f"Unknown cond_select_strategy: {strategy}")

    def _get_cond_ids(self, frame_list_file) -> list:
        """List of 4-digit ids to attempt SAM3D gen on."""
        strategy = self.seq_config["cond_select_strategy"]
        if strategy == "manual":
            return [f"{self.seq_config['cond_idx']:04d}"]
        if strategy == "auto":
            with open(frame_list_file, "r") as f:
                return [f"{int(line.strip()):04d}" for line in f if line.strip()]
        raise ValueError(f"Unknown cond_select_strategy: {strategy}")

    # -----------------------------------------------------------------------
    # Data read / convert
    # -----------------------------------------------------------------------

    def ZED_read_data(self, *_, **__):
        self.print_header("ZED read data")
        _shell("cd /usr/local/zed/tools && ./ZED_Explorer", check=False)

    def ZED_parse_data(self, scene_name, **kwargs):
        """Decode the per-seq .svo2 next to $DATASET_DIR into rgb/ir/meta/depth_ZED."""
        self.print_header(f"ZED parse data for {scene_name}")
        scene_dir = f"{self.dataset_dir}/{scene_name}"
        svo_file = f"{self.dataset_dir}/{scene_name}.svo2"
        os.makedirs(scene_dir, exist_ok=True)
        if self.rebuild:
            _shell(f"cd {shlex.quote(scene_dir)} && rm -rf depth_zed rgb meta ir")
        downsample = int(kwargs.get("downsample", 3))
        # libstdc++ preload works around the pyzed wheel loading the wrong libstdc++.
        self.envs.main.run(
            [
                "svo_export.py",
                "--mode", "2",
                "--input_svo_file", svo_file,
                "--output_path_dir", f"{scene_dir}/",
                "--interval", downsample,
                "--resize_width", "1.0",
                "--resize_height", "1.0",
                "--crop_width", "960",
                "--crop_height", "720",
            ],
            cwd=f"{vggt_code_dir}/third_party/zed-sdk/recording/export/svo/python",
            env_vars={"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"},
        )

    def convert_depth_to_ply(self, scene_name, **kwargs):
        """ZED-only: dump a subset of frames as coloured point clouds for visual QA."""
        self.print_header(f"convert depth to ply for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        depth_dir = "depth_ZED" if "zed" in dataset_type else "depth"
        out_dir = f"ply_{dataset_type}"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(f'{data_dir}/{out_dir}')}")
        self.envs.main.run(
            [
                "depth_to_ply.py",
                "--input_dir", data_dir,
                "--depth_dir", depth_dir,
                "--output_dir", out_dir,
                "--mask_dir", "mask_object",
                "--ply_interval", "10",
                "--use_rgb",
                "--mask_depth",
            ],
            cwd=vggt_code_dir,
        )

    def soft_link_depth(self, scene_name, **__):
        """Publish depth_fs (FoundationStereo output) as the canonical `depth/` dir.
        Only datasets whose raw depth source is FS (ZED / xper1m) need this step —
        ho3d already has per-frame depth/."""
        if "zed" not in dataset_type and dataset_type != "xper1m":
            print(f"soft_link_depth skipped for dataset_type={dataset_type}.")
            return
        self.print_header(f"soft link depth for {scene_name}")
        scene_dir = f"{self.dataset_dir}/{scene_name}"
        _shell(f"cd {shlex.quote(scene_dir)} && rm -rf depth && ln -s depth_fs depth")

    def get_depth_from_foundation_stereo(self, scene_name, **__):
        """FoundationStereo on the left/right IR pair — outputs depth_fs/ + ply_fs/.
        Works for any dataset that laid out `ir/{idx:04d}_{left,right}.png` +
        `meta/{idx:04d}.pkl` with `stereo_camMat` / `stereo_baseline` (currently
        ZED* and xper1m)."""
        if "zed" not in dataset_type and dataset_type != "xper1m":
            print(f"FoundationStereo step skipped for dataset_type={dataset_type}.")
            return
        self.print_header(f"get depth from foundation stereo for {scene_name}")
        scene_dir = f"{self.dataset_dir}/{scene_name}"
        if self.rebuild:
            _shell(f"cd {shlex.quote(scene_dir)} && rm -rf depth_fs ply_fs")
        self.envs.fs.run(
            [
                "scripts/run_video.py",
                "--left_dir", f"{scene_dir}/ir/",
                "--right_dir", f"{scene_dir}/ir/",
                "--intrinsic_file", f"{scene_dir}/meta/0000.pkl",
                "--ckpt_dir", "./pretrained_models/model_best_bp2.pth",
                "--out_dir", f"{scene_dir}/depth_fs/",
                "--ply_dir", f"{scene_dir}/ply_fs/",
                "--ply_interval", "10",
            ],
            cwd=f"{vggt_code_dir}/third_party/FoundationStereo",
        )

    # -----------------------------------------------------------------------
    # xper1m (xperience-10m) raw-episode -> ho3d-style layout
    # -----------------------------------------------------------------------

    def xper1m_convert(self, scene_name, **kwargs):
        """Extract one xperience-10m episode into rgb/ + ir/ + meta/ + mask_object/
        + sam3_prompts/hand.json. Raw episode dir is resolved from `scene_name`
        by sequence_config_xper1m (parses `xp__{uuid}__{ep}`). Subsequent FS /
        SAM3 / HAMER / fit_hand steps all read the standard layout.

        Default mode (XPER1M_HAND_VISIBLE_START=1, the cluster default) scans
        body_mocap/keypoints[19] (R_WRIST in SLAM world) to find the first
        contiguous run of N visible frames, uses those as the extraction
        window, and writes a projected bbox to meta/{idx}.pkl::hand_bbox_2d_right
        plus sam3_prompts/hand.json (box + text=right_hand) for SAM3 to consume."""
        if dataset_type != "xper1m":
            print(f"xper1m_convert only applies to DATASET=xper1m, got {dataset_type}.")
            return
        self.print_header(f"xper1m convert for {scene_name}")
        from confs.sequence_config_xper1m import raw_episode_dir
        raw_ep = raw_episode_dir(scene_name)
        out_dir = f"{self.dataset_dir}/{scene_name}"
        num_frames = int(kwargs.get("num_frames", self.seq_config.get("frame_number", 50)))
        start = int(kwargs.get("start", self.seq_config.get("frame_star", 0)))
        interval = int(kwargs.get("interval", self.seq_config.get("frame_interval", 1)))
        hand_visible = os.getenv("XPER1M_HAND_VISIBLE_START", "1") == "1"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        args = [
            "generator/scripts/xper1m_convert.py",
            "--raw_ep_dir", raw_ep,
            "--out_dir", out_dir,
            "--interval", interval,
            "--num_frames", num_frames,
        ]
        if hand_visible:
            args.append("--start_by_hand_visible")
        else:
            args += ["--start", start]
        # xper1m_convert exits 2 ("no visible-hand run") → CondaEnv.run raises
        # SystemExit → run() loop skips remaining stages for this seq.
        self.envs.main.run(args, cwd=vggt_code_dir)

    # -----------------------------------------------------------------------
    # SAM3 masks (hand + object)
    # -----------------------------------------------------------------------

    def _hand_mask_paths(self, scene_name):
        data_dir = f"{self.dataset_dir}/{scene_name}/rgb/"
        out_mask_dir = f"{self.dataset_dir}/{scene_name}/mask_hand"
        prompt_file = f"{self.dataset_dir}/{scene_name}/sam3_prompts/hand.json"
        return data_dir, out_mask_dir, prompt_file

    _OBJ_MASK_TEXT_PROMPTS = {
        "AP": "blue pitcher base",
        "MPM": "potted meatal can",
        "SB": "white clean bleach",
        "SM": "yellow mustard bottle",
        "ABF": "white clean bleach",
        "BB": "yello banana",
        "GPMF": "potted meatal can",
        "GSF": "scissors",
        "MC": "red cracker_box",
        "MDF": "orange power drill",
        "ND": "orange power drill",
        "SMu": "red mug",
        "SS": "yellow sugar box",
        "ShSu": "yellow sugar box",
        "SiBF": "yellow banana",
        "SiS": "yellow sugar box",
    }

    def _obj_mask_paths(self, scene_name):
        data_dir = f"{self.dataset_dir}/{scene_name}/rgb/"
        out_mask_dir = f"{self.dataset_dir}/{scene_name}/mask_object"
        prompt_file = f"{self.dataset_dir}/{scene_name}/sam3_prompts/object.json"
        obj_family = scene_name.rstrip("0123456789")
        text_prompt = self._OBJ_MASK_TEXT_PROMPTS.get(obj_family, None)
        return data_dir, out_mask_dir, prompt_file, text_prompt

    def _sam3_prompt_batch(self, target, scene_names):
        """Shared logic for hand/object prompt batching. Loads SAM3 model ONCE,
        iterates all seqs with pre-filled popup if a prompt JSON already exists.
        Idempotent — --rebuild does NOT wipe prompt JSON (it's user input)."""
        video_paths, prompt_files, titles, text_prompts = [], [], [], []
        for s in scene_names:
            if target == "hand":
                data_dir, _, pf = self._hand_mask_paths(s)
                tp = "right_hand"
            else:
                data_dir, _, pf, tp = self._obj_mask_paths(s)
                tp = tp if tp else "None"   # literal "None" -> no text prompt in sam3
            video_paths.append(data_dir)
            prompt_files.append(pf)
            titles.append(f"{target} mask — {s}")
            text_prompts.append(tp)
        self.envs.sam3.run(
            [
                "run_HO3D_video.py",
                "--prompt_only",
                "--video_path", *video_paths,
                "--prompt_file", *prompt_files,
                "--prompt_title", *titles,
                "--text_prompt", *text_prompts,
            ],
            cwd=f"{vggt_code_dir}/third_party/sam3/",
        )

    def ho3d_get_hand_mask_prompt(self, scene_name, **__):
        """Click-to-save first-frame hand-mask prompts. Collapses the outer
        per-seq loop into ONE SAM3 invocation (single predictor load).
        Pre-fills popup from existing prompt_file on re-run."""
        if getattr(self, "_hand_prompt_batch_done", False):
            return
        self.print_header(f"ho3d get hand mask PROMPT ({len(self.seq_list)} seq(s))")
        self._sam3_prompt_batch("hand", self.seq_list)
        self._hand_prompt_batch_done = True

    def ho3d_get_obj_mask_prompt(self, scene_name, **__):
        """Object mask prompts. Per-seq text_prompt varies (see
        `_OBJ_MASK_TEXT_PROMPTS`); otherwise identical to the hand variant."""
        if getattr(self, "_obj_prompt_batch_done", False):
            return
        self.print_header(f"ho3d get object mask PROMPT ({len(self.seq_list)} seq(s))")
        self._sam3_prompt_batch("object", self.seq_list)
        self._obj_prompt_batch_done = True

    def _sam3_run_inference(self, scene_name, which):
        """SAM3 full-video mask propagation. `which` ∈ {"hand", "object"}.

        Interactive review (check_mask_result=1) fires a GUI popup so the user
        can edit the first-frame mask. Suppressed when a saved prompt JSON
        exists OR when SAM3_FORCE_NON_INTERACTIVE=1 is set (batch/headless jobs
        like xper1m rely on the text prompt alone)."""
        if which == "hand":
            data_dir, out_mask_dir, prompt_file = self._hand_mask_paths(scene_name)
            text_prompt = "right_hand"
        else:
            data_dir, out_mask_dir, prompt_file, text_prompt = self._obj_mask_paths(scene_name)
            text_prompt = text_prompt or "None"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_mask_dir)}")
        has_prompt = os.path.exists(prompt_file)
        force_non_interactive = os.getenv("SAM3_FORCE_NON_INTERACTIVE", "0") == "1"
        interactive = not (has_prompt or force_non_interactive)
        self.envs.sam3.run(
            [
                "run_HO3D_video.py",
                "--video_path", data_dir,
                "--out_path", out_mask_dir,
                "--text_prompt", text_prompt,
                "--prompt_file", prompt_file,
                "--check_mask_result", "1" if interactive else "0",
            ],
            cwd=f"{vggt_code_dir}/third_party/sam3/",
        )

    def ho3d_get_hand_mask(self, scene_name, **__):
        self.print_header(f"ho3d get hand mask for {scene_name}")
        self._sam3_run_inference(scene_name, "hand")

    def ho3d_get_obj_mask(self, scene_name, **__):
        self.print_header(f"ho3d get object mask for {scene_name}")
        self._sam3_run_inference(scene_name, "object")

    # -----------------------------------------------------------------------
    # HAMER (hand pose)
    # -----------------------------------------------------------------------

    def ho3d_estimate_hand_pose(self, scene_name, **__):
        self.print_header(f"estimate hand pose for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{data_dir}/hands/"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        self.envs.hamer.run(
            [
                "demo.py",
                "--img_folder", f"{data_dir}/rgb",
                "--out_folder", out_dir,
                "--seq_name", scene_name,
                "--full_frame",
                "--body_detector", "wilor_yolo",
            ],
            cwd=f"{vggt_code_dir}/third_party/hamer",
        )

    def ho3d_interpolate_hamer(self, scene_name, **__):
        self.print_header(f"interpolate HAMER results for {scene_name}")
        hands_dir = f"{self.dataset_dir}/{scene_name}/hands"
        self.envs.main.run(
            ["./interpolate.py", "--dataset_dir", hands_dir, "--out_dir", hands_dir],
            cwd=vggt_code_dir,
        )

    # -----------------------------------------------------------------------
    # MANO fit (intrinsic / trans)
    # -----------------------------------------------------------------------

    def _fit_hand(self, scene_name, mode, stage_desc, **kwargs):
        self.print_header(f"fit hand to {stage_desc} for {scene_name}")
        output_dir = f"{self.dataset_dir}/{scene_name}/"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(f'{output_dir}/mano_fit_ckpt/{mode}')}")
        args = [
            "scripts/fit_hand.py",
            "--seq_name", scene_name,
            "--mode", mode,
            "--data_dir", f"{self.dataset_dir}/{scene_name}",
            "--out_dir", output_dir,
        ]
        if "num_frames" in kwargs:
            args += ["--num_frames", kwargs["num_frames"]]
        self.envs.main.run(args, cwd=f"{vggt_code_dir}/generator")

    def fit_hand_intrinsic(self, scene_name, **kwargs):
        self._fit_hand(scene_name, "h_intrinsic", "intrinsic", **kwargs)

    def fit_hand_trans(self, scene_name, **kwargs):
        self._fit_hand(scene_name, "h_trans", "trans", **kwargs)

    def _fit_hand_vis(self, scene_name, mode):
        self.print_header(f"visualize hand fit ({mode}) for {scene_name}")
        self.envs.main.run(
            [
                "scripts/fit_hand_vis.py",
                "--data_dir", f"{self.dataset_dir}/{scene_name}",
                "--mode", mode,
            ],
            cwd=f"{vggt_code_dir}/generator",
        )

    def fit_hand_intrinsic_vis(self, scene_name, **__):
        self._fit_hand_vis(scene_name, "h_intrinsic")

    def fit_hand_trans_vis(self, scene_name, **__):
        self._fit_hand_vis(scene_name, "h_trans")

    # -----------------------------------------------------------------------
    # SAM3D: filter_2D -> gen -> filter_3D -> align (mask/pts/fp) -> align_filter
    # -----------------------------------------------------------------------

    def _sam3d_filter_vis(self, scene_name, frame_list_file):
        self.envs.main.run(
            [
                f"{vggt_code_dir}/robust_hoi_pipeline/pipeline_sam3d_filter_vis.py",
                "--dataset_dir", self.dataset_dir,
                "--scene_name", scene_name,
                "--frame_list_file", frame_list_file,
            ],
        )

    def ho3d_obj_SAM3D_filter_2D(self, scene_name, **__):
        self.print_header(f"SAM3D filter 2D for {scene_name}")
        if self.vis:
            self._sam3d_filter_vis(
                scene_name,
                f"{self.dataset_dir}/{scene_name}/SAM3D/frame_list_after_ftp_filtered.txt",
            )
            return
        self.envs.main.run(
            [
                f"{vggt_code_dir}/robust_hoi_pipeline/pipeline_sam3d_filter_2D.py",
                "--dataset_dir", self.dataset_dir,
                "--scene_name", scene_name,
                "--frame_start", self.seq_config.get("frame_star", 0),
                "--frame_end", self.seq_config.get("frame_end", -1),
                "--frame_interval", self.seq_config.get("frame_interval", 5),
                "--cond_idx", self.seq_config.get("cond_idx", 0),
            ],
        )

    def ho3d_obj_SAM3D_filter_3D(self, scene_name, **__):
        self.print_header(f"SAM3D filter 3D for {scene_name}")
        if self.vis:
            self._sam3d_filter_vis(
                scene_name,
                f"{self.dataset_dir}/{scene_name}/SAM3D/frame_list_after_3d_filtered.txt",
            )
            return
        self.envs.main.run(
            [
                f"{vggt_code_dir}/robust_hoi_pipeline/pipeline_sam3d_filter_3D.py",
                "--dataset_dir", self.dataset_dir,
                "--scene_name", scene_name,
                "--cond_idx", self.seq_config.get("cond_idx", 0),
            ],
        )

    def ho3d_obj_SAM3D_gen(self, scene_name, **__):
        """Batch-generate SAM3D mesh + renders for every candidate cond frame."""
        self.print_header(f"SAM3D gen for {scene_name}")
        frame_list_file = f"{self.dataset_dir}/{scene_name}/SAM3D/frame_list_after_ftp_filtered.txt"
        image_ids = self._get_cond_ids(frame_list_file)
        print(f"SAM3D gen candidate ids: {image_ids}")

        batch_entries = []
        for fid in image_ids:
            out_dir = f"{self.dataset_dir}/{scene_name}/SAM3D/{fid}/"
            if self.rebuild:
                _shell(f"rm -rf {shlex.quote(out_dir)}/*", check=False)
            if os.path.exists(f"{out_dir}/scene.glb"):
                print(f"  skip {fid}: scene.glb already present.")
                continue
            batch_entries.append({
                "image_path": f"{self.dataset_dir}/{scene_name}/rgb/{fid}.jpg",
                "mask_path": f"{self.dataset_dir}/{scene_name}/mask_object/{fid}.png",
                "depth_file": f"{self.dataset_dir}/{scene_name}/depth/{fid}.png",
                "meta_file": f"{self.dataset_dir}/{scene_name}/meta/{fid}.pkl",
                "out_dir": out_dir,
            })

        if not batch_entries:
            print("All candidate frames already have scene.glb; skipping SAM3D gen.")
            return

        batch_file = os.path.join(os.path.dirname(frame_list_file), "sam3d_gen_batch.json")
        with open(batch_file, "w") as f:
            json.dump(batch_entries, f, indent=2)
        print(f"Wrote {len(batch_entries)} entries to {batch_file}")

        args = ["demo.py", "--batch-file", batch_file]
        if self.vis:
            args.append("--vis")
        self.envs.sam3d.run(
            args,
            cwd=f"{vggt_code_dir}/third_party/sam-3d-objects",
            env_vars={"LIDRA_SKIP_INIT": "1"},
        )

    def _sam3d_aligned_vis(self, scene_name, method, frame_list_file=None, frame_indices=None):
        args = [
            f"{vggt_code_dir}/robust_hoi_pipeline/pipeline_sam3d_aligned_vis.py",
            "--dataset_dir", self.dataset_dir,
            "--scene_name", scene_name,
            "--align_method", method,
        ]
        if frame_list_file is not None:
            args += ["--frame_list_file", frame_list_file]
        if frame_indices is not None:
            args += ["--frame_indices", frame_indices]
        self.envs.main.run(args)

    def ho3d_align_SAM3D_mask(self, scene_name, **__):
        self.print_header(f"align SAM3D (mask) for {scene_name}")
        if self.vis:
            self._sam3d_aligned_vis(scene_name, "mask")
            return
        frame_list_file = f"{self.dataset_dir}/{scene_name}/SAM3D/frame_list_after_3d_filtered.txt"
        with open(frame_list_file, "r") as f:
            image_ids = [f"{int(line.strip()):04d}" for line in f if line.strip()]
        for fid in image_ids:
            out_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_mask/{fid}/"
            if self.rebuild:
                _shell(f"rm -rf {shlex.quote(out_dir)}/*", check=False)
            self.envs.main.run(
                [
                    "align_SAM3D_mask.py",
                    "--data-dir", f"{self.dataset_dir}/{scene_name}/",
                    "--hand-pose-suffix", "trans",
                    "--cond-index", int(fid),
                    "--out-dir", out_dir,
                ],
                cwd=vggt_code_dir,
            )

    def ho3d_align_SAM3D_pts(self, scene_name, **__):
        """Score each candidate mesh by the mean 3D-track projection error.
        Writes scores.json + frame_list_after_aligned_pts.txt (sorted)."""
        self.print_header(f"align SAM3D (pts) for {scene_name}")
        if self.vis:
            self._sam3d_aligned_vis(scene_name, "pts")
            return
        frame_list_file = f"{self.dataset_dir}/{scene_name}/SAM3D/frame_list_after_3d_filtered.txt"
        with open(frame_list_file, "r") as f:
            image_ids = [f"{int(line.strip()):04d}" for line in f if line.strip()]
        scores = {}
        for fid in image_ids:
            out_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_pts/{fid}/"
            if self.rebuild:
                _shell(f"rm -rf {shlex.quote(out_dir)}/*", check=False)
            self.envs.main.run(
                [
                    "align_SAM3D_pts.py",
                    "--data-dir", f"{self.dataset_dir}/{scene_name}/",
                    "--cond-index", int(fid),
                    "--SAM3D-index", int(fid),
                    "--out-dir", out_dir,
                ],
                cwd=vggt_code_dir,
            )
            alignment_json = os.path.join(out_dir, "alignment.json")
            if os.path.exists(alignment_json):
                with open(alignment_json, "r") as f:
                    scores[fid] = json.load(f).get("mean_error", float("inf"))
            else:
                scores[fid] = float("inf")
            print(f"  id={fid} mean_error={scores[fid]:.4f}")

        sorted_scores = dict(sorted(scores.items(), key=lambda kv: kv[1]))
        out_root = f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_pts"
        with open(f"{out_root}/scores.json", "w") as f:
            json.dump(sorted_scores, f, indent=2)
        with open(f"{out_root}/frame_list_after_aligned_pts.txt", "w") as f:
            for sid in sorted_scores:
                f.write(f"{sid}\n")
        print("Ranking (best -> worst):")
        for rank, (sid, err) in enumerate(sorted_scores.items(), 1):
            print(f"  {rank}. id={sid} mean_error={err:.4f}")

    def ho3d_align_SAM3D_fp(self, scene_name, **__):
        """Refine each candidate mesh's pose via FoundationPose."""
        self.print_header(f"align SAM3D (fp) for {scene_name}")
        if self.vis:
            self._sam3d_aligned_vis(
                scene_name,
                "fp",
                frame_list_file=f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_fp/frame_list_after_aligned_fp.txt",
            )
            return
        out_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_fp/"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}/*", check=False)
        # Note: this script lives under vggt (not FoundationPose), but it imports
        # FoundationPose modules, so the cwd must be FoundationPose for relative
        # configs/assets lookups to resolve.
        self.envs.main.run(
            [
                f"{vggt_code_dir}/align_SAM3D_fp.py",
                "--data_dir", f"{self.dataset_dir}/{scene_name}",
                "--out_dir", out_dir,
            ],
            cwd=f"{vggt_code_dir}/third_party/FoundationPose",
        )

    def ho3d_align_by_foundation_pose(self, scene_name, **__):
        """Standalone (non-filter) FoundationPose alignment on the best cond frame."""
        self.print_header(f"align by FoundationPose for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        best_id = self._get_best_cond_id(scene_name)
        out_dir = f"{data_dir}/foundation_pose_align/{best_id}/"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        args = [
            f"{vggt_code_dir}/align_by_foundation_pose.py",
            "--data_dir", data_dir,
            "--cond_index", int(best_id),
            "--out_dir", out_dir,
        ]
        if self.vis:
            args.append("--vis_in_rerun")
        self.envs.fp.run(args, cwd=f"{vggt_code_dir}/third_party/FoundationPose")

    def pipeline_sam3d_align_filter(self, scene_name, **__):
        """Filter aligned candidate frames by depth 3-axis coverage."""
        out_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_align_filter/"
        self.print_header(f"SAM3D align-filter coverage for {scene_name}")
        if self.vis:
            self._sam3d_aligned_vis(
                scene_name,
                "fp",
                frame_list_file=f"{out_dir}/frame_list_align_filter.txt",
            )
            return
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}/*", check=False)
        self.envs.main.run(
            [
                f"{vggt_code_dir}/robust_hoi_pipeline/pipeline_sam3d_align_filter.py",
                "--dataset_dir", self.dataset_dir,
                "--scene_name", scene_name,
                "--out_dir", out_dir,
                "--cond_idx", self.seq_config.get("cond_idx", 0),
            ],
        )

    def pipeline_sam3d_delete_unused(self, scene_name, **__):
        self.print_header(f"delete unused SAM3D folders for {scene_name}")
        self.envs.main.run(
            [
                f"{vggt_code_dir}/robust_hoi_pipeline/pipeline_sam3d_delete_unused.py",
                "--dataset_dir", self.dataset_dir,
                "--scene_name", scene_name,
                "--cond_idx", self.seq_config.get("cond_idx", 0),
            ],
        )

    def pipeline_sam3d_best_id(self, scene_name, **__):
        """Pick the single best cond frame. `--vis` shows best_id in Rerun."""
        if self.vis:
            self.print_header(f"visualize best_id for {scene_name}")
            extra = {}
            if self.seq_config["cond_select_strategy"] == "manual":
                extra["frame_indices"] = self._get_best_cond_id(scene_name)
            self._sam3d_aligned_vis(
                scene_name,
                "fp",
                frame_list_file=f"{self.dataset_dir}/{scene_name}/SAM3D_align_filter/best_id.txt",
                **extra,
            )
            return
        self.print_header(f"find best id for {scene_name}")
        self.envs.main.run(
            [
                f"{vggt_code_dir}/robust_hoi_pipeline/pipeline_sam3d_best_id.py",
                "--scene_dir", f"{self.dataset_dir}/{scene_name}",
            ],
        )

    def pipeline_sam3d_best_id_sum(self, scene_name, **__):
        """Cross-sequence best_id summary (uses all seqs defined in the config)."""
        self.print_header("summarize best id across all sequences")
        output_file = f"{vggt_code_dir}/output/metrics_summary/best_id_sum.txt"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(output_file)}")
        self.envs.main.run(
            [
                f"{vggt_code_dir}/robust_hoi_pipeline/pipeline_sam3d_best_id_sum.py",
                "--dataset_dir", self.dataset_dir,
                "--sequence_list", ",".join(sequence_name_list),
                "--output_file", output_file,
            ],
        )

    def ho3d_SAM3D_post_process(self, scene_name, **__):
        """Finalise best SAM3D mesh into SAM3D_aligned_post_process/<best_id>/."""
        self.print_header(f"SAM3D post-process for {scene_name}")
        best_id = self._get_best_cond_id(scene_name)
        src_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_fp/{best_id}/"
        sam3d_dir = f"{self.dataset_dir}/{scene_name}/SAM3D/{best_id}/"
        dst_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_post_process/{best_id}/"
        if self.vis:
            self.envs.main.run(
                ["SAM3D_post_process_vis.py", "--out-dir", dst_dir],
                cwd=vggt_code_dir,
            )
            return
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(dst_dir)}")
        self.envs.main.run(
            [
                "SAM3D_post_process.py",
                "--src-dir", src_dir,
                "--sam3d-dir", sam3d_dir,
                "--dst-dir", dst_dir,
            ],
            cwd=vggt_code_dir,
        )

    # -----------------------------------------------------------------------
    # Legacy align_gen_3d (VGGT-based, kept per run_wonder_hoi.sh)
    # -----------------------------------------------------------------------

    def _cond_index_div_interval(self, scene_name):
        """Condition index in interval-downsampled frame indexing."""
        best = int(self._get_best_cond_id(scene_name))
        return best // self.seq_config["frame_interval"]

    def ho3d_align_gen_3d(self, scene_name, **__):
        self.print_header(f"align gen 3D for {scene_name}")
        data_dir = f"{vggt_code_dir}/output/{scene_name}/"
        out_dir = f"{vggt_code_dir}/output/{scene_name}/gen_3d_aligned/"
        if self.vis:
            self.envs.main.run(
                [
                    "viewer/viewer_step.py",
                    "--result_folder", f"{out_dir}/../results/",
                    "--vis_only_register",
                    "--vis_only_keyframes",
                    "--num_frames", self.seq_config["frame_number"],
                    "--log_aligned_mesh", "1",
                ],
                cwd=vggt_code_dir,
            )
            return
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}/*", check=False)
        self.envs.main.run(
            [
                "align_gen_3d.py",
                "--input_dir", data_dir,
                "--output_dir", out_dir,
                "--init_pose_image_idx", self._cond_index_div_interval(scene_name),
            ],
            cwd=vggt_code_dir,
        )

    def ho3d_align_gen_3d_omni(self, scene_name, **__):
        self.print_header(f"align gen 3D omni for {scene_name}")
        keyframe_dir = f"{vggt_code_dir}/output/{scene_name}/results/"
        gen3d_aligned_dir = f"{vggt_code_dir}/output/{scene_name}/gen_3d_aligned/refined/"
        out_dir = f"{vggt_code_dir}/output/{scene_name}/gen_3d_aligned_omni/"
        if self.vis:
            self.envs.main.run(
                [
                    "viewer/viewer_step.py",
                    "--result_folder", f"{out_dir}/../results/",
                    "--vis_only_register",
                    "--vis_only_keyframes",
                    "--num_frames", self.seq_config["frame_number"],
                    "--log_aligned_mesh", "1",
                ],
                cwd=vggt_code_dir,
            )
            return
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}/*", check=False)
        self.envs.main.run(
            [
                "align_gen_3d_omni.py",
                "--keyframe_dir", keyframe_dir,
                "--gen3d_aligned_dir", gen3d_aligned_dir,
                "--output_dir", out_dir,
            ],
            cwd=vggt_code_dir,
        )

    def ho3d_obj_sdf_optimization(self, scene_name, **__):
        """instant-nsr-pl launch with extra compiler env for CUDA JIT kernels."""
        self.print_header(f"ho3d object sdf optimization for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/"
        result_dir = f"{vggt_code_dir}/output/{scene_name}/results/"
        out_dir = f"{vggt_code_dir}/output/{scene_name}/sdf_optimization/"
        best_id = self._get_best_cond_id(scene_name)
        self.envs.main.run(
            [
                "launch.py",
                "--config", "configs/neus-mixed.yaml",
                "--train",
                f"dataset.root_dir={result_dir}",
                f"dataset.sam3d_root_dir={data_dir}/SAM3D_aligned_post_process/{best_id}/",
                "--exp_dir", out_dir,
            ],
            cwd=f"{vggt_code_dir}/third_party/instant-nsr-pl",
            env_vars={"CC": "gcc-11", "CXX": "g++-11", "CUDAHOSTCXX": "g++-11"},
        )

    # -----------------------------------------------------------------------
    # ho3d evaluation helpers
    # -----------------------------------------------------------------------

    def _ho3d_eval(self, scene_name, hand_fit_mode):
        self.print_header(f"vggt ho3d evaluate ({hand_fit_mode}) for {scene_name}")
        result_dir = f"{vggt_code_dir}/output/{scene_name}/results/"
        out_dir = f"{vggt_code_dir}/output/{scene_name}/eval_{hand_fit_mode}/"
        if self.vis:
            self.envs.main.run(
                ["viewer_pose.py", "--result_folder", result_dir],
                cwd=vggt_code_dir,
            )
            return
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}/*", check=False)
        self.envs.main.run(
            [
                "eval.py",
                "--result_folder", result_dir,
                "--out_dir", out_dir,
                "--hand_fit_mode", hand_fit_mode,
            ],
            cwd=vggt_code_dir,
        )

    def ho3d_eval_intrinsic(self, scene_name, **__):
        self._ho3d_eval(scene_name, "intrinsic")

    def ho3d_eval_trans(self, scene_name, **__):
        self._ho3d_eval(scene_name, "trans")

    def ho3d_eval_rot(self, scene_name, **__):
        self._ho3d_eval(scene_name, "rot")

    # -----------------------------------------------------------------------
    # HOI pipeline (data preprocess -> corres -> joint_opt -> align_hand_object)
    # -----------------------------------------------------------------------

    def hoi_pipeline_data_preprocess(self, scene_name, **__):
        self.print_header(f"hoi pipeline data preprocess for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{data_dir}/pipeline_preprocess/"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        self.envs.main.run(
            [
                "robust_hoi_pipeline/pipeline_data_preprocess.py",
                "--data_dir", data_dir,
                "--output_dir", out_dir,
                "--start", "0",
                "--end", self.seq_config["frame_end"],
                "--interval", self.seq_config["frame_interval"],
                "--cond_index", int(self._get_best_cond_id(scene_name)),
            ],
            cwd=vggt_code_dir,
        )

    def _neus_init_cmd(self, scene_name, out_dir, max_steps, robust_hoi_weight,
                       sam3d_weight, export_only=False, **kwargs):
        """Shared shell for pipeline_neus_init.py — max_steps / loss weights
        differ between data_preprocess_sam3d_neus / neus_init / neus_global."""
        data_dir = f"{self.dataset_dir}/{scene_name}"
        result_dir = f"{vggt_code_dir}/output/{scene_name}/"
        args = [
            "robust_hoi_pipeline/pipeline_neus_init.py",
            "--data_dir", data_dir,
            "--output_dir", out_dir,
            "--result_dir", f"{result_dir}/",
            "--cond_index", int(self._get_best_cond_id(scene_name)),
            "--max_steps", max_steps,
            "--robust_hoi_weight", robust_hoi_weight,
            "--sam3d_weight", sam3d_weight,
        ]
        if "max_registered_frames" in kwargs:
            args += ["--max_registered_frames", int(kwargs["max_registered_frames"])]
        if export_only:
            args.append("--export_only")
        conda_prefix = self.envs.main.root
        shell_prefix = (
            f"export PATH={conda_prefix}/bin:$PATH "
            f"&& export CC={conda_prefix}/bin/x86_64-conda-linux-gnu-gcc "
            f"&& export CXX={conda_prefix}/bin/x86_64-conda-linux-gnu-g++"
        )
        self.envs.main.run(args, cwd=vggt_code_dir, shell_prefix=shell_prefix)

    def hoi_pipeline_data_preprocess_sam3d_neus(self, scene_name, **kwargs):
        """Short (3k step) NeuS init from SAM3D alone (no robust HOI) — used as
        a warm-up / sanity-check before the full pipeline_neus_init."""
        self.print_header(f"hoi pipeline sam3d neus init for {scene_name}")
        best_id = self._get_best_cond_id(scene_name)
        out_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_post_process/{best_id}/neus/"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        self._neus_init_cmd(scene_name, out_dir, 3000, 0.0, 1.0, **kwargs)

    def hoi_pipeline_neus_init(self, scene_name, **kwargs):
        """Full NeuS init (10k steps, robust HOI loss on)."""
        self.print_header(f"hoi pipeline neus init for {scene_name}")
        out_dir = f"{vggt_code_dir}/output/{scene_name}/pipeline_neus_init"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        self._neus_init_cmd(scene_name, out_dir, 10000, 1.0, 0.0, **kwargs)

    def hoi_pipeline_neus_global(self, scene_name, **kwargs):
        """NeuS global stage (2k refinement). `--export_only true` skips training."""
        self.print_header(f"hoi pipeline neus global for {scene_name}")
        out_dir = f"{vggt_code_dir}/output/{scene_name}/pipeline_neus_global"
        export_only = str(kwargs.get("export_only", "")).lower() in {"1", "true", "yes"}
        if self.rebuild and not export_only:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        self._neus_init_cmd(scene_name, out_dir, 2000, 1.0, 0.0, export_only=export_only, **kwargs)

    def hoi_pipeline_get_corres(self, scene_name, **__):
        self.print_header(f"hoi pipeline get correspondences for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/pipeline_preprocess"
        out_dir = f"{self.dataset_dir}/{scene_name}/pipeline_corres"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        self.envs.main.run(
            [
                "robust_hoi_pipeline/pipeline_get_corres.py",
                "--data_dir", data_dir,
                "--out_dir", out_dir,
                "--cond_index", int(self._get_best_cond_id(scene_name)),
            ],
            cwd=vggt_code_dir,
        )

    def hoi_pipeline_eval_corres(self, scene_name, **__):
        self.print_header(f"hoi pipeline evaluate correspondences for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/pipeline_preprocess"
        corres_dir = f"{self.dataset_dir}/{scene_name}/pipeline_corres"
        out_dir = f"{corres_dir}/eval_corres_vis"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        self.envs.main.run(
            [
                "robust_hoi_pipeline/pipeline_eval_corres.py",
                "--data_dir", data_dir,
                "--corres_dir", corres_dir,
                "--out_dir", out_dir,
                "--cond_index", "850",  # hard-coded in original; leaving for parity
            ],
            cwd=vggt_code_dir,
        )

    def hoi_pipeline_joint_opt(self, scene_name, **__):
        """Joint optimization of obj pose + hand + NeuS prior. `--vis` renders
        overlays via `pipeline_joint_opt_vis.py` instead."""
        self.print_header(f"hoi pipeline joint optimization for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{vggt_code_dir}/output/{scene_name}"
        if self.vis:
            args = [
                "robust_hoi_pipeline/pipeline_joint_opt_vis.py",
                "--data_dir", data_dir,
                "--output_dir", out_dir,
                "--cond_index", int(self._get_best_cond_id(scene_name)),
            ]
            if dataset_type != "ho3d":
                args += ["--vis_gt", "0"]
            self.envs.main.run(args, cwd=vggt_code_dir)
            return
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}/pipeline_joint_opt")
        self.envs.main.run(
            [
                "robust_hoi_pipeline/pipeline_joint_opt.py",
                "--data_dir", data_dir,
                "--output_dir", out_dir,
                "--cond_index", int(self._get_best_cond_id(scene_name)),
                "--optimize_3D_prior",
            ],
            cwd=vggt_code_dir,
        )

    def hoi_pipeline_joint_opt_global(self, scene_name, **__):
        self.print_header(f"hoi pipeline global joint optimization for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{vggt_code_dir}/output/{scene_name}"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}/pipeline_joint_opt/9999")
        self.envs.main.run(
            [
                "robust_hoi_pipeline/pipeline_joint_opt_global.py",
                "--result_dir", out_dir,
                "--data_dir", data_dir,
            ],
            cwd=vggt_code_dir,
        )

    def hoi_pipeline_reg_remaining(self, scene_name, **__):
        self.print_header(f"hoi pipeline register remaining for {scene_name}")
        self.envs.main.run(
            [
                "robust_hoi_pipeline/pipeline_reg_remaining.py",
                "--data_dir", f"{self.dataset_dir}/{scene_name}",
                "--output_dir", f"{vggt_code_dir}/output/{scene_name}",
                "--cond_index", int(self._get_best_cond_id(scene_name)),
            ],
            cwd=vggt_code_dir,
        )

    def _align_hand_object(self, scene_name, mode):
        """Hand-object alignment. `mode` ∈ {"h","r","o","ho"} picks which
        sub-MANO is frozen/free during optimization."""
        data_dir = f"{self.dataset_dir}/{scene_name}"
        result_dir = f"{vggt_code_dir}/output/{scene_name}/pipeline_joint_opt/"
        out_dir = f"{vggt_code_dir}/output/{scene_name}/align_hand_object"

        if self.vis:
            self.print_header(f"align hand-object ({mode}) vis for {scene_name}")
            vis_out_dir = f"{result_dir}/eval_vis_{mode}/"
            if self.rebuild:
                _shell(f"rm -rf {shlex.quote(vis_out_dir)}")
            self.envs.main.run(
                [
                    "robust_hoi_pipeline/pipeline_joint_opt_eval_vis_nvdiffrast.py",
                    "--result_folder", result_dir,
                    "--out_dir", vis_out_dir,
                    "--SAM3D_dir", f"{data_dir}/SAM3D_aligned_post_process",
                    "--cond_index", int(self._get_best_cond_id(scene_name)),
                    "--hand_mode", mode,
                    "--render_hand",
                    "--vis_gt", "0",
                ],
                cwd=vggt_code_dir,
            )
            return

        if self.rebuild:
            _shell(
                f"rm -rf {shlex.quote(f'{out_dir}/hold_fit.aligned_{mode}.npy')} "
                f"&& rm -rf {shlex.quote(f'{out_dir}/mano_fit_ckpt/{mode}/')}"
            )
        self.envs.main.run(
            [
                "scripts/align_hands_object.py",
                "--seq_name", scene_name,
                "--mode", mode,
                "--out_dir", out_dir,
                "--data_dir", data_dir,
                "--result_dir", result_dir,
            ],
            cwd=f"{vggt_code_dir}/generator",
        )

    def hoi_pipeline_align_hand_object_h(self, scene_name, **__):
        self._align_hand_object(scene_name, "h")

    def hoi_pipeline_align_hand_object_r(self, scene_name, **__):
        self._align_hand_object(scene_name, "r")

    def hoi_pipeline_align_hand_object_o(self, scene_name, **__):
        self._align_hand_object(scene_name, "o")

    def hoi_pipeline_align_hand_object_ho(self, scene_name, **__):
        self._align_hand_object(scene_name, "ho")

    def hoi_pipeline_eval(self, scene_name, **__):
        self.print_header(f"hoi pipeline eval for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{vggt_code_dir}/output/{scene_name}"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}/pipeline_joint_opt/eval")
        self.envs.main.run(
            [
                "robust_hoi_pipeline/pipeline_joint_opt_eval.py",
                "--result_folder", f"{out_dir}/pipeline_joint_opt/",
                "--out_dir", f"{out_dir}/pipeline_joint_opt/eval/",
                "--SAM3D_dir", f"{data_dir}/SAM3D_aligned_post_process",
                "--cond_index", int(self._get_best_cond_id(scene_name)),
            ],
            cwd=vggt_code_dir,
        )

    def hoi_pipeline_eval_vis(self, scene_name, **__):
        """nvdiffrast overlay: RGB + predicted obj mesh + hand (mode=ho)."""
        self.print_header(f"hoi pipeline eval vis for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        result_dir = f"{vggt_code_dir}/output/{scene_name}/pipeline_joint_opt/"
        out_dir = f"{result_dir}/eval_vis/"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        args = [
            "robust_hoi_pipeline/pipeline_joint_opt_eval_vis_nvdiffrast.py",
            "--result_folder", result_dir,
            "--out_dir", out_dir,
            "--SAM3D_dir", f"{data_dir}/SAM3D_aligned_post_process",
            "--cond_index", int(self._get_best_cond_id(scene_name)),
            "--hand_mode", "ho",
            "--render_hand",
        ]
        if dataset_type != "ho3d":
            args += ["--vis_gt", "0"]
        self.envs.main.run(args, cwd=vggt_code_dir)

    def hoi_pipeline_eval_vis_gt(self, scene_name, **__):
        self.print_header(f"visualize GT mesh/pose in Rerun for {scene_name}")
        self.envs.main.run(
            [
                "robust_hoi_pipeline/pipeline_joint_opt_eval_vis_gt.py",
                "--data_dir", f"{self.dataset_dir}/{scene_name}",
                "--render_hand",
            ],
            cwd=vggt_code_dir,
        )

    def hoi_pipeline_teaser(self, scene_name, **__):
        self.print_header(f"hoi pipeline teaser for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        result_dir = f"{vggt_code_dir}/output/{scene_name}/pipeline_joint_opt/"
        out_dir = f"{result_dir}/teaser/"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        self.envs.main.run(
            [
                "robust_hoi_pipeline/pipeline_teaser.py",
                "--result_folder", result_dir,
                "--SAM3D_dir", f"{data_dir}/SAM3D_aligned_post_process",
                "--cond_index", int(self._get_best_cond_id(scene_name)),
                "--out_dir", out_dir,
                "--hand_mode", "ho",
                "--render_hand",
                "--mesh_type", self.seq_config.get("teaser_mesh_type", "neus"),
                "--start", self.seq_config.get("teaser_start", 100),
                "--end", self.seq_config.get("teaser_end", 1000),
                "--interval", self.seq_config.get("teaser_interval", 100),
            ],
            cwd=vggt_code_dir,
        )

    # -----------------------------------------------------------------------
    # Metrics summary
    # -----------------------------------------------------------------------

    def _eval_sum(self, scene_name, fit_mode):
        self.print_header(f"eval summary ({fit_mode!r}) for {scene_name}")
        output_file = f"{vggt_code_dir}/output/metrics_summary/eval{fit_mode}.txt"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(output_file)}")
        self.envs.main.run(
            [
                "extract_jsons.py",
                "--parent_dir", "output",
                "--metric_folder", f"eval{fit_mode}",
                "--output_file", output_file,
            ],
            cwd=vggt_code_dir,
        )

    def eval_sum_intrinsic(self, scene_name, **__):
        self._eval_sum(scene_name, "_intrinsic")

    def eval_sum_trans(self, scene_name, **__):
        self._eval_sum(scene_name, "_trans")

    def eval_sum_rot(self, scene_name, **__):
        self._eval_sum(scene_name, "_rot")

    def eval_sum(self, scene_name, **__):
        self._eval_sum(scene_name, "")

    def eval_sum_vis(self, scene_name, **kwargs):
        """Grid video comparing ours vs the baselines (+ GT when ho3d)."""
        self.print_header(f"eval summary vis for {scene_name}")
        baseline_root = f"{vggt_code_dir}/output_baseline/{scene_name}"
        defaults = {
            "foundation_dir": f"{baseline_root}/foundation_sam3d",
            "bundle_sdf_dir": f"{baseline_root}/bundle_sdf",
            "hold_dir": f"{baseline_root}/hold",
            "joint_opt_dir": f"{vggt_code_dir}/output/{scene_name}/pipeline_joint_opt/eval_vis",
            "gt_dir": f"{baseline_root}/gt",
            "out_dir": f"{vggt_code_dir}/output/metrics_summary/{scene_name}/",
        }
        paths = {k: kwargs.get(k, v) for k, v in defaults.items()}
        if self.rebuild:
            _shell(
                f"rm -rf {shlex.quote(paths['out_dir'])} "
                f"&& rm -f {shlex.quote(f'{vggt_code_dir}/output/metrics_summary/eval_sum_{scene_name}.mp4')}"
            )
        args = [
            "robust_hoi_pipeline/eval_sum_vis.py",
            "--foundation_dir", paths["foundation_dir"],
            "--bundle_sdf_dir", paths["bundle_sdf_dir"],
            "--hold_dir", paths["hold_dir"],
            "--joint_opt_dir", paths["joint_opt_dir"],
            "--gt_dir", paths["gt_dir"],
            "--out_dir", paths["out_dir"],
        ]
        for key in ("fps", "line_width", "line_gray"):
            if key in kwargs:
                args += [f"--{key}", kwargs[key]]
        if "vis_method_name" in kwargs:
            flag = str(kwargs["vis_method_name"]).lower() in {"1", "true", "yes", "y"}
            args.append("--vis_method_name" if flag else "--no_vis_method_name")
        if self.rebuild:
            args.append("--rebuild")
        self.envs.main.run(args, cwd=vggt_code_dir)

    # -----------------------------------------------------------------------
    # Baseline re-visualisation (fp / bundlesdf / hold / gt)
    # -----------------------------------------------------------------------

    def foundation_pose_eval_vis(self, scene_name, **__):
        self.print_header(f"foundation_pose eval vis for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        result_folder = f"{vggt_code_dir}/third_party/FoundationPose/output/sam3d/{scene_name}"
        out_dir = f"{vggt_code_dir}/output_baseline/{scene_name}/foundation_sam3d"
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        args = [
            "third_party/FoundationPose/eval_vis_nvdiffrast.py",
            "--result_folder", result_folder,
            "--data_dir", data_dir,
            "--sam3d_dir", f"{data_dir}/SAM3D",
            "--out_dir", out_dir,
            "--cond_index", int(self.seq_config["cond_idx"]),
        ]
        if dataset_type != "ho3d":
            args += ["--vis_gt", "0"]
        self.envs.main.run(args, cwd=vggt_code_dir)

    def _baseline_mesh_vis(self, scene_name, script, output_root, data_root, vis_root, **kwargs):
        """Shared shell for bundle_sdf / hold baselines — both take the same
        --seq_list/--output_root/--data_root/--vis_root/--fps/--alpha signature."""
        fps = kwargs.get("fps", 6)
        alpha = kwargs.get("alpha", 0.8)
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(vis_root)}")
        args = [
            script,
            "--seq_list", scene_name,
            "--output_root", output_root,
            "--data_root", data_root,
            "--vis_root", vis_root,
            "--fps", fps,
            "--alpha", alpha,
        ]
        return args

    def bundle_sdf_eval_vis(self, scene_name, **kwargs):
        self.print_header(f"bundle_sdf eval vis for {scene_name}")
        args = self._baseline_mesh_vis(
            scene_name,
            "third_party/bundlesdf/eval_vis_nvdiffrast.py",
            kwargs.get("output_root", f"{vggt_code_dir}/third_party/bundlesdf/output_ho3d"),
            kwargs.get("data_root", self.dataset_dir),
            kwargs.get("vis_root", f"{vggt_code_dir}/output_baseline/{scene_name}/bundle_sdf/"),
            **kwargs,
        )
        if dataset_type != "ho3d":
            args += ["--vis_gt", "0"]
        self.envs.main.run(args, cwd=vggt_code_dir)

    def hold_eval_vis(self, scene_name, **kwargs):
        self.print_header(f"hold eval vis for {scene_name}")
        args = self._baseline_mesh_vis(
            scene_name,
            "third_party/hold/code/eval_vis_nvdiffrast.py",
            kwargs.get("output_root", f"{vggt_code_dir}/third_party/hold/code/logs_ho3d"),
            kwargs.get("data_root", f"{vggt_code_dir}/third_party/hold/code/data_ho3d"),
            kwargs.get("vis_root", f"{vggt_code_dir}/output_baseline/{scene_name}/hold/"),
            **kwargs,
        )
        args += ["--vis_gt", "0"]
        self.envs.main.run(args, cwd=vggt_code_dir)

    def gt_eval_vis(self, scene_name, **kwargs):
        self.print_header(f"gt eval vis for {scene_name}")
        out_dir = kwargs.get("out_dir", f"{vggt_code_dir}/output_baseline/{scene_name}/gt/")
        if self.rebuild:
            _shell(f"rm -rf {shlex.quote(out_dir)}")
        args = [
            "robust_hoi_pipeline/eval_gt_vis.py",
            "--data_dir", f"{self.dataset_dir}/{scene_name}",
            "--out_dir", out_dir,
        ]
        if str(kwargs.get("render_hand", "false")).lower() in {"1", "true", "yes", "y"}:
            args.append("--render_hand")
        for key in ("fps", "alpha", "max_frames"):
            if key in kwargs:
                args += [f"--{key}", kwargs[key]]
        if self.rebuild:
            args.append("--rebuild")
        self.envs.main.run(args, cwd=vggt_code_dir)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


# Keep in sync with the methods registered in `run_wonder_hoi.process_mapping`.
# argparse `choices` needs a flat list of strings.
_PROCESS_CHOICES = [
    # data_read
    "ZED_read_data",
    # data_convert
    "ZED_parse_data", "convert_depth_to_ply", "soft_link_depth",
    "get_depth_from_foundation_stereo",
    "ho3d_get_hand_mask_prompt", "ho3d_get_obj_mask_prompt",
    "ho3d_get_hand_mask", "ho3d_get_obj_mask",
    "ho3d_estimate_hand_pose", "ho3d_interpolate_hamer",
    "xper1m_convert",
    # obj_process
    "ho3d_obj_SAM3D_filter_2D", "ho3d_obj_SAM3D_filter_3D", "ho3d_obj_SAM3D_gen",
    "ho3d_align_SAM3D_mask", "ho3d_align_SAM3D_pts", "ho3d_align_SAM3D_fp",
    "ho3d_align_by_foundation_pose",
    "pipeline_sam3d_align_filter", "pipeline_sam3d_delete_unused",
    "pipeline_sam3d_best_id", "pipeline_sam3d_best_id_sum",
    "ho3d_SAM3D_post_process",
    "ho3d_align_gen_3d", "ho3d_align_gen_3d_omni", "ho3d_obj_sdf_optimization",
    "ho3d_eval_intrinsic", "ho3d_eval_trans", "ho3d_eval_rot",
    "hoi_pipeline_data_preprocess", "hoi_pipeline_data_preprocess_sam3d_neus",
    "hoi_pipeline_get_corres", "hoi_pipeline_eval_corres",
    "hoi_pipeline_neus_init", "hoi_pipeline_neus_global",
    "hoi_pipeline_joint_opt", "hoi_pipeline_joint_opt_global",
    "hoi_pipeline_reg_remaining",
    "hoi_pipeline_align_hand_object_h", "hoi_pipeline_align_hand_object_r",
    "hoi_pipeline_align_hand_object_o", "hoi_pipeline_align_hand_object_ho",
    "hoi_pipeline_eval", "hoi_pipeline_eval_vis", "hoi_pipeline_eval_vis_gt",
    "hoi_pipeline_teaser",
    "eval_sum_intrinsic", "eval_sum_trans", "eval_sum_rot", "eval_sum", "eval_sum_vis",
    # hand_pose_postprocess
    "fit_hand_intrinsic", "fit_hand_trans",
    "fit_hand_intrinsic_vis", "fit_hand_trans_vis",
    # baseline
    "foundation_pose_eval_vis", "bundle_sdf_eval_vis", "hold_eval_vis", "gt_eval_vis",
]


def main(args, extras):
    """Fold unknown `--key value` pairs into a dict; ints/floats are parsed."""
    extras_dict = {}
    for i in range(0, len(extras), 2):
        if i + 1 >= len(extras):
            break
        key = extras[i].lstrip("-")
        value = extras[i + 1]
        try:
            value = float(value) if "." in value else int(value)
        except ValueError:
            pass
        extras_dict[key] = value
    run_wonder_hoi(args, extras_dict).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq_list",
        nargs="+",
        required=True,
        help="Sequence name(s). Use 'all' to expand to sequence_name_list.",
    )
    parser.add_argument(
        "--execute_list",
        choices=[
            "data_read",
            "data_convert",
            "obj_process",
            "hand_pose_postprocess",
            "baseline",
        ],
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--process_list",
        choices=_PROCESS_CHOICES,
        nargs="+",
        required=True,
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--reconstruction_dir",
        type=str,
        default=f"{vggt_code_dir}/output_backup",
        help="Reconstruction folder (rarely overridden).",
    )
    args, extras = parser.parse_known_args()
    if "all" in args.seq_list:
        args.seq_list = sequence_name_list
    main(args, extras)
