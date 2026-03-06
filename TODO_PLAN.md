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
- [ ] Priority 1
- [ ] Priority 2
- [ ] Priority 3

#### Tasks
- [ ] Task A
- [ ] Task B
- [ ] Task C

#### Follow-ups / Blockers
- [ ] Follow-up 1
- [ ] Blocker 1

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
