import os
import shutil
import argparse
import json
import time
import random
import numpy as np
from PIL import Image

from confs.sequence_config import sequences, vggt_code_dir, home_dir, conda_dir


class run_wonder_hoi:
    def __init__(self, args, extras):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.code_dir = os.path.join(self.current_dir, "..")
        self.seq_list = args.seq_list
        self.execute_list = args.execute_list
        self.process_list = args.process_list
        self.dataset_dir = args.dataset_dir
        self.reconstruction_dir = args.reconstruction_dir
        self.rebuild = args.rebuild
        self.vis = args.vis
        self.eval = args.eval
        
        self.conda_type = args.conda_type # "anaconda3" or "miniconda3"
        self.conda_dir = conda_dir
        if self.conda_type is not None:
            self.conda_dir = f"{home_dir}/{self.conda_type}"
        elif not os.path.exists(self.conda_dir):
            if os.path.exists(f"{home_dir}/anaconda3"):
                self.conda_dir = f"{home_dir}/anaconda3"
            else:
                self.conda_dir = f"{home_dir}/miniconda3"
        self.extras = extras
        self.process_mapping = {
            "data_read": {
                "realsense_read_data": self.realsense_read_data,
                "ZED_read_data": self.ZED_read_data,
            },
            "data_convert": {
                "ZED_parse_data": self.ZED_parse_data,
                "convert_depth_to_ply": self.convert_depth_to_ply,
                "get_depth_from_foundation_stereo": self.get_depth_from_foundation_stereo,
                "hot3d_cp_images": self.hot3d_cp_images,
                "hot3d_gen_meta": self.hot3d_gen_meta,
                "hot3d_sync_to_local": self.hot3d_sync_to_local,
                "hot3d_sync_hands_to_local": self.hot3d_sync_hands_to_local,
                "hot3d_get_undistorted_stereo": self.hot3d_get_undistorted_stereo,
                "hot3d_get_pose_in_cam": self.hot3d_get_pose_in_cam,
                "hot3d_validate_pose_in_cam": self.hot3d_validate_pose_in_cam,
                "hot3d_get_depth": self.hot3d_get_depth,
                "hot3d_image_to_video": self.hot3d_image_to_video,
                "hot3d_get_mask": self.hot3d_get_mask,
                "hot3d_estimate_hand_pose": self.hot3d_estimate_hand_pose,
                "hot3d_interpolate_hamer": self.hot3d_interpolate_hamer,
                "hot3d_mask_only_object": self.hot3d_mask_only_object,
                "hot3d_convert_to_stereo": self.hot3d_convert_to_stereo,
                "zed_sync_data_to_local": self.zed_sync_data_to_local,
                "ho3d_estimate_hand_pose": self.ho3d_estimate_hand_pose,
                "ho3d_interpolate_hamer": self.ho3d_interpolate_hamer,
                "ho3d_get_hand_mask": self.ho3d_get_hand_mask,
                "ho3d_get_obj_mask": self.ho3d_get_obj_mask,
                "ho3d_inpaint": self.ho3d_inpaint,
            },
            "obj_process": {
                "estimate_obj_pose": self.estimate_obj_pose,
                "obj_3D_gen": self.obj_3D_gen,
                "align_mesh_image": self.align_mesh_image,
                "get_pts_observed": self.get_pts_observed,
                "align_corres": self.align_corres,
                "align_pcs": self.align_pcs,
                "scale_3D_gen": self.scale_3D_gen,
                "mesh2SDF": self.mesh2SDF,
                "hunyuan_omni": self.hunyuan_omni,
                "hot3d_obj_pose": self.hot3d_obj_pose,
                "zed_joint_optimization": self.zed_joint_optimization,
                "zed_sync_data_to_local": self.zed_sync_data_to_local,
                "ho3d_obj_3D_gen": self.ho3d_obj_3D_gen,
                "ho3d_condition_id": self.ho3d_condition_id,
                "ho3d_obj_SAM3D_gen": self.ho3d_obj_SAM3D_gen,
                "ho3d_obj_SAM3D_post_opt_GS": self.ho3d_obj_SAM3D_post_opt_GS,
                "ho3d_obj_SAM3D_post_optimization": self.ho3d_obj_SAM3D_post_optimization,
                "ho3d_align_SAM3D_mask": self.ho3d_align_SAM3D_mask,
                "ho3d_align_SAM3D_pts": self.ho3d_align_SAM3D_pts,
                "ho3d_SAM3D_post_process": self.ho3d_SAM3D_post_process,
                "ho3d_keyframe_optimization": self.ho3d_keyframe_optimization,
                "ho3d_align_gen_3d": self.ho3d_align_gen_3d,
                "ho3d_align_gen_3d_omni": self.ho3d_align_gen_3d_omni,
                "ho3d_obj_sdf_optimization": self.ho3d_obj_sdf_optimization,
                "ho3d_eval_intrinsic": self.ho3d_eval_intrinsic,
                "ho3d_eval_trans": self.ho3d_eval_trans,
                "ho3d_eval_rot": self.ho3d_eval_rot,
                "eval_sum_intrinsic": self.eval_sum_intrinsic,
                "eval_sum_trans": self.eval_sum_trans,
                "eval_sum_rot": self.eval_sum_rot,
                "eval_sum": self.eval_sum,
                "hoi_pipeline_neus_init": self.hoi_pipeline_neus_init,
                "hoi_pipeline_data_preprocess": self.hoi_pipeline_data_preprocess,
                "hoi_pipeline_data_preprocess_sam3d_neus": self.hoi_pipeline_data_preprocess_sam3d_neus,
                "hoi_pipeline_get_corres": self.hoi_pipeline_get_corres,
                "hoi_pipeline_align_SAM3D_with_HY": self.hoi_pipeline_align_SAM3D_with_HY,
                "hoi_pipeline_3D_points_align_with_HY": self.hoi_pipeline_3D_points_align_with_HY,
                "hoi_pipeline_HY_to_SAM3D": self.hoi_pipeline_HY_to_SAM3D,
                "hoi_pipeline_joint_opt": self.hoi_pipeline_joint_opt,
                "hoi_pipeline_reg_remaining": self.hoi_pipeline_reg_remaining,
                "hoi_pipeline_HY_gen": self.hoi_pipeline_HY_gen,
                "hoi_pipeline_HY_omni_gen": self.hoi_pipeline_HY_omni_gen,
            },
            "hand_pose_preprocess": {
                "estimate_hand_pose": self.estimate_hand_pose,
            },
            "hand_pose_postprocess": {
                "fit_hand_intrinsic": self.fit_hand_intrinsic,
                "fit_hand_trans": self.fit_hand_trans,
                "fit_hand_rot": self.fit_hand_rot,
                "fit_hand_pose": self.fit_hand_pose,
                "fit_hand_all": self.fit_hand_all,
                "fit_hand_viewer": self.fit_hand_viewer,
            },
        }

    def run(self):
        for exe in self.execute_list:
            for process in self.process_list:
                for seq in self.seq_list:
                    self.seq_config = sequences["default"]
                    if seq in sequences:
                        # TODO: override default config with specific sequence config
                        for key, value in sequences[seq].items():
                            self.seq_config[key] = value

                    self.process_mapping[exe][process](seq, **self.extras)


    def images_to_video(self, search_pattern, output_video_path, fps=30, rebuild=True):
        if rebuild:
            cmd = f"rm -rf {output_video_path}"
            print(cmd)
            os.system(cmd)

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        cmd = ""
        cmd += f"/usr/bin/ffmpeg -y -i {search_pattern} "
        cmd += f"-framerate {fps} -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "
        cmd += f'''-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '''
        cmd += f"{output_video_path}"
        print(cmd)
        os.system(cmd)

    def hot3d_obj_pose(self, scene_name, **kwargs):
        self.print_header(f"hot3d estimate object pose for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/processed/"
        # obj_ids = [3, 4, 5, 6, 7, 8]
        obj_ids = [5]
        for obj_id in obj_ids:
            out_dir = f"{home_dir}/Documents/project/mpsfm/outputs/{scene_name}/{obj_id}/"
            instance = self.seq_config["obj_id_to_instance"][obj_id].split("_")[-1]
            if self.rebuild:
                ### remmeber to remove the previous results
                cmd = f"rm -rf {out_dir}"
                print(cmd)
                os.system(cmd)

            if self.vis:
                cmd = f"cd {home_dir}/Documents/project/mpsfm && "
                cmd += f"{self.conda_dir}/envs/mpsfm/bin/python scripts/rerun_vis.py "
                cmd += f"--data_dir {data_dir} "
                cmd += f"--out_dir {out_dir} "
                cmd += f"--sequence_name {scene_name} "
                cmd += f"--aligned_transform_json {out_dir}/aligned/0200.json " # hard code
                cmd += f"--gt_obj_path {self.dataset_dir}/assets/{instance}.glb " # hard code
                cmd += f"--gen_obj_path {data_dir}/3D_gen/cube_asset/white_mesh_remesh.obj "
                cmd += f"--mode refined gt "
                print(cmd)
                os.system(cmd)
                return

            cmd = f"cd {home_dir}/Documents/project/mpsfm && "
            cmd += f"{self.conda_dir}/envs/mpsfm/bin/python run.py "
            cmd += f"--data_dir {self.dataset_dir}/{scene_name}/processed "
            cmd += f"--out_dir {out_dir} "
            cmd += f"--start {self.seq_config['frame_star']} "
            cmd += f"--end {self.seq_config['frame_end']} "
            cmd += f"--interval 1 "
            cmd += f"--config_name sp-lg_depthfd_m3dv2 "
            cmd += f"--verbose 2 "
            cmd += f"--limit 100 "
            cmd += f"--obj_id {obj_id} "
            cmd += f"--obj_instance {instance} "
            print(cmd)
            os.system(cmd)
        
    def hot3d_get_depth(self, scene_name, **kwargs):
        self.print_header(f"hot3d get depth for {scene_name}")
        pair = "214-1_1201-2"
        data_dir = f"{self.dataset_dir}/{scene_name}/processed/"

        if self.rebuild:
            cmd = f"rm -rf {self.dataset_dir}/{scene_name}/processed/depth_fs/ {self.dataset_dir}/{scene_name}/processed/ply_fs/"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/FoundationStereo && "
        cmd += f"{self.conda_dir}/envs/foundation_stereo/bin/python "
        cmd += f"scripts/hot3d_stereo.py "
        cmd += f"--data_dir {data_dir} "
        cmd += f"--ckpt_dir ./pretrained_models/model_best_bp2.pth "
        cmd += f"--out_dir {self.dataset_dir}/{scene_name}/processed/depth_fs/ "
        cmd += f"--ply_dir {self.dataset_dir}/{scene_name}/processed/ply_fs/ "
        cmd += f"--ply_interval 10 "
        # cmd += f"--denoise_cloud "
        # cmd += f"--realsense "
        # cmd += f"--visualize_cloud "
        print(cmd)
        os.system(cmd)

    def hot3d_image_to_video(self, scene_name, **kwargs):
        data_dir = f"{self.dataset_dir}/{scene_name}/processed/images"
        out_video = f"{data_dir}/left.mp4"

        if self.rebuild:
            cmd = f"rm -rf {out_video}"
            print(cmd)
            os.system(cmd)

        if not os.path.exists(out_video):
            self.images_to_video(f"{data_dir}/%04d.png", out_video, fps=5, rebuild=self.rebuild)

    def hot3d_get_mask(self, scene_name, **kwargs):
        self.print_header(f"hot3d get mask for {scene_name}")

        data_dir = f"{self.dataset_dir}/{scene_name}/processed/images/"
        out_mask_dir = f"{self.dataset_dir}/{scene_name}/processed/masks"
        out_mask_rgba_dir = f"{self.dataset_dir}/{scene_name}/processed/images_RGBA"
        cutie_workspace = f"{home_dir}/Documents/project/Cutie/workspace/{scene_name}_hot3d"

        if self.rebuild:
            cmd = f"rm -rf {out_mask_dir} && "
            cmd += f"rm -rf {out_mask_rgba_dir} && "
            cmd += f"rm -rf {cutie_workspace} "
            print(cmd)
            os.system(cmd)
        
        # if not os.path.exists(data_dir+"/left.mp4"):
        #     self.hot3d_image_to_video(scene_name, **kwargs)
        cmd = f"cd {home_dir}/Documents/project/Cutie/ && "
        cmd += f"{self.conda_dir}/envs/py38cu118/bin/python interactive_demo.py "
        cmd += f"--images {data_dir} "
        cmd += f"--workspace {cutie_workspace} "
        cmd += f"--num_objects {2 + self.seq_config['obj_num']}" # 2 for hands
        print(cmd)
        os.system(cmd)

        # copy generated masks and images
        cmd = f"cp -rf {cutie_workspace}/masks {out_mask_dir}"
        print(cmd)
        os.system(cmd)
        cmd = f"cp -rf {cutie_workspace}/images_RGBA {out_mask_rgba_dir} "
        print(cmd)
        os.system(cmd)

    def ho3d_image_to_video(self, data_dir, **kwargs):
        out_video = f"{data_dir}/left.mp4"

        if self.rebuild:
            cmd = f"rm -rf {out_video}"
            print(cmd)
            os.system(cmd)

        if not os.path.exists(out_video):
            self.images_to_video(f"{data_dir}/%04d.jpg", out_video, fps=5, rebuild=self.rebuild)

    def ho3d_get_hand_mask_1(self, scene_name, **kwargs):
        self.print_header(f"ho3d get mask for {scene_name}")

        data_dir = f"{self.dataset_dir}/train/{scene_name}/rgb/"
        out_mask_dir = f"{self.dataset_dir}/train/{scene_name}/mask_hand"
        cutie_workspace = f"{home_dir}/Documents/project/Cutie/workspace/{scene_name}_ho3d"

        if self.rebuild:
            cmd = f"rm -rf {out_mask_dir} && "
            cmd += f"rm -rf {cutie_workspace} "
            print(cmd)
            os.system(cmd)
        
        if not os.path.exists(data_dir+"/left.mp4"):
            self.ho3d_image_to_video(data_dir, **kwargs)
        
        cmd = f"cd {home_dir}/Documents/project/Cutie/ && "
        cmd += f"{self.conda_dir}/envs/py38cu118/bin/python interactive_demo.py "
        cmd += f"--images {data_dir} "
        cmd += f"--video {data_dir}/left.mp4 "
        cmd += f"--workspace {cutie_workspace} "
        cmd += f"--num_objects 1 " # 2 for hands
        print(cmd)
        os.system(cmd)

        # copy generated masks and images
        cmd = f"cp -rf {cutie_workspace}/masks/* {out_mask_dir}"
        print(cmd)
        os.system(cmd)

    def ho3d_get_hand_mask(self, scene_name, **kwargs):
        self.print_header(f"ho3d get mask for {scene_name}")

        data_dir = f"{self.dataset_dir}/{scene_name}/rgb/"
        out_mask_dir = f"{self.dataset_dir}/{scene_name}/mask_hand"
        text_prompt = "right_hand"


        if self.rebuild:
            cmd = f"rm -rf {out_mask_dir}"
            print(cmd)
            os.system(cmd)
        
        
        cmd = f"cd {home_dir}/Documents/project/sam3/ && "
        cmd += f"{self.conda_dir}/envs/sam3/bin/python run_HO3D_video.py "
        cmd += f"--video_path {data_dir} "
        cmd += f"--out_path {out_mask_dir} "
        cmd += f"--text_prompt '{text_prompt}' "
        # cmd += f"--use_both_text_and_point_prompt "
        print(cmd)
        os.system(cmd)

    def ho3d_get_obj_mask(self, scene_name, **kwargs):
        self.print_header(f"ho3d get mask for {scene_name}")

        data_dir = f"{self.dataset_dir}/{scene_name}/rgb/"
        out_mask_dir = f"{self.dataset_dir}/{scene_name}/mask_object"
        obj_name = scene_name.rstrip("0123456789")


        obj2text_prompt = {
            'AP': 'blue pitcher base',
            'MPM': 'potted meatal can',
            'SB': 'white clean bleach',
            'SM': 'yellow mustard bottle',
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

        prompt_text_str = obj2text_prompt[obj_name]

        if self.rebuild:
            cmd = f"rm -rf {out_mask_dir}"
            print(cmd)
            os.system(cmd)
        
        
        cmd = f"cd {home_dir}/Documents/project/sam3/ && "
        cmd += f"{self.conda_dir}/envs/sam3/bin/python run_HO3D_video.py "
        cmd += f"--video_path {data_dir} "
        cmd += f"--out_path {out_mask_dir} "
        cmd += f"--text_prompt '{prompt_text_str}' "
        # cmd += f"--use_both_text_and_point_prompt "
        print(cmd)
        os.system(cmd)        
    
     

    def hot3d_estimate_hand_pose(self, scene_name, **kwargs):
        self.print_header(f"estimate hand pose for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/processed/"
        out_dir = f"{data_dir}/hands/"

        if self.rebuild:
            ### remmeber to remove the previous results
            cmd = f"rm -rf {out_dir}"
            print(cmd)
            os.system(cmd)



        # hamer inference

        cmd = f"cd {self.code_dir}/generator/hamer && "
        cmd += f"{self.conda_dir}/envs/hamer/bin/python demo.py "
        cmd += f"--img_folder {data_dir}/images "
        cmd += f"--out_folder {out_dir} "
        cmd += f"--seq_name {scene_name} "
        cmd += f"--full_frame "
        cmd += f"--body_detector wilor_yolo "
        # cmd += f"--side_view "
        # cmd += f"--save_mesh "
        # cmd += f"--save_render "
        print(cmd)
        os.system(cmd)


    def ho3d_estimate_hand_pose(self, scene_name, **kwargs):
        self.print_header(f"estimate hand pose for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{data_dir}/hands/"

        if self.rebuild:
            ### remmeber to remove the previous results
            cmd = f"rm -rf {out_dir}"
            print(cmd)
            os.system(cmd)



        # hamer inference

        cmd = f"cd {self.code_dir}/generator/hamer && "
        cmd += f"{self.conda_dir}/envs/hamer/bin/python demo.py "
        cmd += f"--img_folder {data_dir}/rgb "
        cmd += f"--out_folder {out_dir} "
        cmd += f"--seq_name {scene_name} "
        cmd += f"--full_frame "
        cmd += f"--body_detector wilor_yolo "
        # cmd += f"--side_view "
        # cmd += f"--save_mesh "
        # cmd += f"--save_render "
        print(cmd)
        os.system(cmd)        

    def ho3d_interpolate_hamer(self, scene_name, **kwargs):
        self.print_header(f"validate hamer results for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/hands"
        out_dir = f"{self.dataset_dir}/{scene_name}/hands"
        data_dir = f"{self.dataset_dir}/{scene_name}/hands"
        out_dir = f"{self.dataset_dir}/{scene_name}/hands"

        cmd = ""
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python ./interpolate.py "
        cmd += f" --dataset_dir {data_dir} "
        cmd += f" --out_dir {out_dir} "
        print(f"{cmd}")
        os.system(cmd)            

    def hot3d_interpolate_hamer(self, scene_name, **kwargs):
        self.print_header(f"validate hamer results for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/processed/hands"
        out_dir = f"{self.dataset_dir}/{scene_name}/processed/hands"

        cmd = ""
        cmd += f" python ./interpolate.py "
        cmd += f" --dataset_dir {data_dir} "
        cmd += f" --out_dir {out_dir} "
        print(f"{cmd}")
        os.system(cmd)      

    def hot3d_mask_only_object(self, scene_name, **kwargs):
        self.print_header(f"hot3d get object mask for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/processed/"
        out_dir = f"{self.dataset_dir}/{scene_name}/processed/images_objects/"

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {self.code_dir}/scripts && "
        cmd += f"python mask_objects.py --data_dir {data_dir} --out_dir {out_dir}"
        print(cmd)
        os.system(cmd)

    def hot3d_convert_to_stereo(self, scene_name, **kwargs):
        self.print_header(f"hot3d convert to stereo for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/"
        out_dir = f"{self.dataset_dir}/{scene_name}/processed/"

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/HOT3D && "
        cmd += f"pixi run viewer --sequence_folder ${data_dir}/${scene_name} "
        cmd += f"--object_library_folder ${data_dir}/assets/ "
        cmd += f"--mano_model_folder ${data_dir}/body_models/ "
        cmd += f"--hand_type MANO "
        cmd += f"--out_dir ${data_dir}/${scene_name}/processed/stereo "
        cmd += f"--headless "


    def fit_hand_viewer(self, scene_name, **kwargs):
        cmd = ""
        cmd += "cd viewer && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python viewer.py "
        cmd += f"--sequence_folder {self.dataset_dir}/{scene_name} "
        cmd += f"--reconstruction_folder {self.reconstruction_dir}/{scene_name} "
        cmd += f"--mano_model_folder {self.code_dir}/body_models/ "
        cmd += f"--rrd_output_path output/{scene_name}/rerun.rrd "
        if "world_coordinate" in kwargs:
            cmd += f"--world_coordinate {kwargs['world_coordinate']} "
        if "only_key_frame" in kwargs:
            if kwargs['only_key_frame'].lower() == "true":
                cmd += f"--only_key_frame "
        if "show_on_mesh_lab" in kwargs:
            if kwargs['show_on_mesh_lab'].lower() == "true":
                cmd += f"--show_on_mesh_lab "
        print(cmd)
        os.system(cmd)

    def _cleanup_mano_ckpt(self, output_dir, mode):
        if not self.rebuild:
            return

        cmd = f"rm -rf {output_dir}/mano_fit_ckpt/{mode}"
        print(cmd)
        os.system(cmd)

    def fit_hand_step(self, scene_name, mode, output_dir, **kwargs):
        frame_number = self.seq_config['frame_number']
        frame_interval = self.seq_config["frame_interval"]

        cmd_parts = [
            f"cd {self.code_dir}/generator &&",
            f"{self.conda_dir}/envs/vggsfm_tmp/bin/python scripts/align_hands_object.py",
            f"--seq_name {scene_name}",
            f"--mode {mode}",
            f"--max_frame_num 9999",
            f"--frame_interval 1",
            f"--out_dir {output_dir}",
        ]
        if "num_frames" in kwargs:
            cmd_parts.append(f"--num_frames {kwargs['num_frames']}")
        if "dataset_type" in kwargs:
            cmd_parts.append(f"--dataset_type {kwargs['dataset_type']}")

        cmd = " ".join(cmd_parts)
        print(cmd)
        os.system(cmd)

    def _fit_hand(self, scene_name, mode, output_dir, stage_desc, **kwargs):
        self.print_header(f"fit hand to {stage_desc} for {scene_name}")
        self._cleanup_mano_ckpt(output_dir, mode)
        self.fit_hand_step(scene_name, mode, output_dir, **kwargs)

    def fit_hand_intrinsic(self, scene_name, **kwargs):
        mode = "h_intrinsic"
        output_dir = f"{self.dataset_dir}/{scene_name}/"
        self._fit_hand(scene_name, mode, output_dir, "intrinsic", **kwargs)

    def fit_hand_trans(self, scene_name, **kwargs):
        mode = "h_trans"
        output_dir = f"{self.dataset_dir}/{scene_name}/"
        self._fit_hand(scene_name, mode, output_dir, "trans", **kwargs)

    def fit_hand_rot(self, scene_name, **kwargs):
        mode = "h_rot"
        output_dir = f"{self.dataset_dir}/{scene_name}/"    
        self._fit_hand(scene_name, mode, output_dir, "rot", **kwargs)

    def fit_hand_pose(self, scene_name, **kwargs):
        mode = "h_pose"
        output_dir = f"{self.dataset_dir}/{scene_name}/"    
        self._fit_hand(scene_name, mode, output_dir, "pose", **kwargs)

    def fit_hand_all(self, scene_name, **kwargs):
        mode = "h_all"
        output_dir = f"{self.dataset_dir}/{scene_name}/"    
        self._fit_hand(scene_name, mode, output_dir, "all", **kwargs)

    def estimate_hand_pose(self, scene_name, **kwargs):
        self.print_header(f"estimate hand pose for {scene_name}")

        if self.rebuild:
            ### remmeber to remove the previous results
            cmd = f"cd {self.dataset_dir}/{scene_name} && rm -rf crop_image mano_fit_ckpt mesh_fit_vis mesh_fit_vis metro_vis 2d_keypoints boxes.npy hpe_vis j2d.crop.npy j2d.full.npy v3d.npy hold_fit.init.npy hold_fit.slerp.npy"
            print(cmd)
            os.system(cmd)

        # ### we first use 100DoH detector instead of hamer inter detector to find hand bounding boxes:
        # cmd = f"cd {self.dataset_dir}/generator/hand_detector.d2 && {self.conda_dir}/envs/hamer/bin/python crop_images.py --scale 1.5 --seq_name {scene_name} --min_size 256 --max_size 700"
        # print(cmd)
        # os.system(cmd)

        # hamer inference

        cmd = f"cd {self.code_dir}/generator/hamer && {self.conda_dir}/envs/hamer/bin/python demo.py --seq_name {scene_name} --batch_size=2  --full_frame --body_detector vitdet"
        print(cmd)
        os.system(cmd)

        ### Register MANO model to predicted meshes (hold_fit.init.npy):
        cmd = f"cd {self.code_dir}/generator && {self.conda_dir}/envs/hamer/bin/python ./scripts/register_mano.py --seq_name {scene_name} --save_mesh --hand_type right --use_beta_loss"
        print(cmd)
        os.system(cmd)

        ### After registeration, run this to linearly interpolate missing frames (hold_fit.slerp.npy):
        cmd = f"cd {self.code_dir}/generator && python ./scripts/validate_hamer.py --seq_name {scene_name}"
        print(cmd)
        os.system(cmd)
    
    def estimate_obj_pose(self, scene_name, **kwargs):
        self.print_header(f"estimate object pose for {scene_name}")

        out_dir = f"{self.code_dir}/code/output/{scene_name}"
        if self.rebuild:
            cmd = f"rm -rf {out_dir}"
            print(cmd)
            os.system(cmd)
        # this command should be run in the docker container threestudio with the env py38
            
        id = f"{self.seq_config['cond_idx']:04d}"

        if self.vis:
            cmd = ""
            cmd += "cd viewer && "
            cmd += f"{self.conda_dir}/envs/threestudio/bin/python viewer.py "
            cmd += f"--sequence_folder {self.dataset_dir}/{scene_name} "
            cmd += f"--cond_id {id} "
            cmd += f"--reconstruction_folder {out_dir} "
            cmd += f"--mano_model_folder {self.code_dir}/body_models/ "
            cmd += f"--rrd_output_path output/{scene_name}/rerun.rrd "
            cmd += f"--world_coordinate object "
            cmd += f"--only_key_frame "
            print(cmd)
            os.system(cmd)
            return
     
        
        output_log_file = f"{out_dir}/{scene_name}/output_full.log"
        if not os.path.exists(output_log_file):
            os.makedirs(f"{out_dir}/{scene_name}", exist_ok=True)
        # cmd = f"cd {self.code_dir}/generator/BundleSDF && {self.conda_dir}/envs/py38/bin/python run_ho3d.py --video_dirs {self.dataset_dir}/{scene_name} --out_dir {out_dir} --scene {scene_name} > {output_log_file}"
        cmd = f"cd {self.code_dir}/generator/BundleSDF && "
        cmd += f"{self.conda_dir}/envs/py38/bin/python run_ho3d.py " 
        cmd += f"--video_dirs {self.dataset_dir}/{scene_name} "
        cmd += f"--out_dir {out_dir} --scene {scene_name} "
        cmd += f"--coarse_align_dir {self.dataset_dir}/{scene_name}/align/{id}.json "
        # cmd += f"> {output_log_file} "
        print(cmd)
        os.system(cmd)

    def obj_3D_gen(self, scene_name, **kwargs):
        self.print_header(f"Generate object 3D model from Hunyuan for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        out_dir = f"{self.dataset_dir}/{scene_name}/3D_gen/{id}/"
                   
        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        # python example.py --image_file $IMAGE_FILE --output_dir $scene_output_dir
        cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-2.1/ && "
        cmd += f"{self.conda_dir}/envs/hunyuan_2.1/bin/python example.py "
        # cmd += f"--image_file {self.dataset_dir}/{scene_name}/mask_obj/{id}.png "
        cmd += f"--image_file {self.dataset_dir}/{scene_name}/processed/inpaint/{id}.png "
        cmd += f"--output_dir {out_dir}"
        print(cmd)
        os.system(cmd)

    def hot3d_obj_3D_gen(self, scene_name, **kwargs):
        self.print_header(f"Generate object 3D model from Hunyuan for {scene_name}")
        # id = f"{self.seq_config['cond_idx']:04d}"
        obj_ids = [5]
        for id in obj_ids:
            out_dir = f"{self.dataset_dir}/{scene_name}/3D_gen/{id:04d}/"
                    
            if self.rebuild:
                cmd = f"rm -rf {out_dir}/*"
                print(cmd)
                os.system(cmd)

            # python example.py --image_file $IMAGE_FILE --output_dir $scene_output_dir
            cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-2.1/ && "
            cmd += f"{self.conda_dir}/envs/hunyuan_2.1/bin/python example.py "
            # cmd += f"--image_file {self.dataset_dir}/{scene_name}/mask_obj/{id}.png "
            cmd += f"--image_file {self.dataset_dir}/{scene_name}/processed/inpaint/{id:04d}.png "
            cmd += f"--output_dir {out_dir}"
            print(cmd)
            os.system(cmd)  

    def ho3d_obj_3D_gen(self, scene_name, **kwargs):
        self.print_header(f"Generate object 3D model from Hunyuan for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        out_dir = f"{self.dataset_dir}/{scene_name}/3D_gen/{id}/"
                   
        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        # python example.py --image_file $IMAGE_FILE --output_dir $scene_output_dir
        cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-2.1/ && "
        cmd += f"{self.conda_dir}/envs/hunyuan_2.1/bin/python example.py "
        # cmd += f"--image_file {self.dataset_dir}/{scene_name}/mask_obj/{id}.png "
        cmd += f"--image_file {self.dataset_dir}/{scene_name}/inpaint/{id}_rgba_center.png "
        cmd += f"--output_dir {out_dir}"
        print(cmd)
        os.system(cmd)

    def ho3d_condition_id(self, scene_name, **kwargs):
        self.print_header(f"ho3d get condition id for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/"
        strategy = "most_common" # bbox or most_common
        out_dir = f"{self.dataset_dir}/{scene_name}/condition_id/{strategy}/"


        if self.rebuild:
            cmd = f"rm -rf {out_dir}"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/vggt/ && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python get_condition_id_{strategy}.py "
        cmd += f"--scene_dir {data_dir} "
        cmd += f"--frame_interval {self.seq_config['frame_interval']} "
        cmd += f"--out_dir {out_dir} "
        print(cmd)
        os.system(cmd)

    def ho3d_obj_SAM3D_gen(self, scene_name, **kwargs):
        self.print_header(f"Generate object 3D model from SAM3D for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        image_path = f"{self.dataset_dir}/{scene_name}/rgb/{id}.jpg"
        depth_path = f"{self.dataset_dir}/{scene_name}/depth/{id}.png"
        mask_path = f"{self.dataset_dir}/{scene_name}/mask_object/{id}.png"
        meta_path = f"{self.dataset_dir}/{scene_name}/meta/{id}.pkl"
        out_dir = f"{self.dataset_dir}/{scene_name}/SAM3D/{id}/"
                   
        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        # python example.py --image_file $IMAGE_FILE --output_dir $scene_output_dir
        cmd = f"cd {home_dir}/Documents/project/sam-3d-objects && "
        cmd += f"LIDRA_SKIP_INIT=1 {self.conda_dir}/envs/sam3d-objects/bin/python demo.py "
        # cmd += f"--image_file {self.dataset_dir}/{scene_name}/mask_obj/{id}.png "
        cmd += f"--image-path {image_path} "
        cmd += f"--mask-path {mask_path} "
        cmd += f"--depth-file {depth_path} "
        cmd += f"--meta-file {meta_path} "
        cmd += f"--out-dir {out_dir} "
        if self.vis:
            cmd += f"--vis "
        
        # cmd = f"cd {home_dir}/Documents/project/sam-3d-objects && "
        # cmd = f"mv {self.dataset_dir}/{scene_name}/SAM3D {self.dataset_dir}/{scene_name}/SAM3D_[no_rendering_optimization] "
        print(cmd)        
        os.system(cmd) 

    def ho3d_obj_SAM3D_post_opt_GS(self, scene_name, **kwargs):
        self.print_header(f"Post-optimize SAM3D Gaussian Splatting for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        out_dir = f"{self.dataset_dir}/{scene_name}/SAM3D/{id}/"

        cmd = f"cd {home_dir}/Documents/project/sam-3d-objects && "
        cmd += f"LIDRA_SKIP_INIT=1 {self.conda_dir}/envs/sam3d-objects/bin/python post_opt_GS.py "
        cmd += f"--out-dir {out_dir} "
        print(cmd)
        os.system(cmd)

    def ho3d_obj_SAM3D_post_optimization(self, scene_name, **kwargs):
        self.print_header(f"Generate object 3D model from SAM3D for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        image_path = f"{self.dataset_dir}/{scene_name}/rgb/{id}.jpg"
        depth_path = f"{self.dataset_dir}/{scene_name}/depth/{id}.png"
        mask_path = f"{self.dataset_dir}/{scene_name}/mask_object/{id}.png"
        meta_path = f"{self.dataset_dir}/{scene_name}/meta/{id}.pkl"


        out_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_optimized/{id}/"
                   
        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        # python example.py --image_file $IMAGE_FILE --output_dir $scene_output_dir
        cmd = f"cd {home_dir}/Documents/project/sam-3d-objects && "
        cmd += f"LIDRA_SKIP_INIT=1 {self.conda_dir}/envs/sam3d-objects/bin/python post_optimization.py "
        # cmd += f"--image_file {self.dataset_dir}/{scene_name}/mask_obj/{id}.png "
        cmd += f"--image-path {image_path} "
        cmd += f"--mask-path {mask_path} "
        cmd += f"--depth-file {depth_path} "
        cmd += f"--meta-file {meta_path} "
        cmd += f"--demo-out-dir {self.dataset_dir}/{scene_name}/SAM3D/{id}/ "
        # cmd += f"--demo-out-dir {self.dataset_dir}/{scene_name}/SAM3D/{id}/ "
        cmd += f"--out-dir {out_dir} "
        # cmd += f"--enable-manual-alignment "
        # cmd += f"--enable-shape-icp "
        cmd += f"--enable-rendering-optimization "
        cmd += f""

        if self.vis:
            cmd += f"--vis "
        
        # cmd = f"cd {home_dir}/Documents/project/sam-3d-objects && "
        # cmd = f"mv {self.dataset_dir}/{scene_name}//SAM3D_[depth_cond][layout_postprocess][ICP_align]/ {self.dataset_dir}/{scene_name}//SAM3D_[depth_cond][ICP_align][mask_align]/ "
        print(cmd)        
        os.system(cmd)         

    def ho3d_align_SAM3D_mask(self, scene_name, **kwargs):
        self.print_header(f"Align SAM3D model for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        image_path = f"{self.dataset_dir}/{scene_name}/rgb/{id}.jpg"
        depth_path = f"{self.dataset_dir}/{scene_name}/depth/{id}.png"
        mask_path = f"{self.dataset_dir}/{scene_name}/mask_object/{id}.png"
        meta_path = f"{self.dataset_dir}/{scene_name}/meta/{id}.pkl"


        out_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_mask/{id}/"
                   
        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        # python example.py --image_file $IMAGE_FILE --output_dir $scene_output_dir
        cmd = f"cd {home_dir}/Documents/project/vggt && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python align_SAM3D_mask.py "
        cmd += f"--data-dir {self.dataset_dir}/{scene_name}/ "
        cmd += f"--hand-pose-suffix trans "
        cmd += f"--cond-index {self.seq_config['cond_idx']} "
        cmd += f"--out-dir {out_dir} "

        if self.vis:
            cmd += f"--vis "

        print(cmd)        
        os.system(cmd)

    def ho3d_align_SAM3D_pts(self, scene_name, **kwargs):
        self.print_header(f"Align SAM3D model using 3D points for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        out_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_pts/{id}/"
                   
        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        # python example.py --image_file $IMAGE_FILE --output_dir $scene_output_dir
        cmd = f"cd {home_dir}/Documents/project/vggt && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python align_SAM3D_pts.py "
        cmd += f"--data-dir {self.dataset_dir}/{scene_name}/ "
        # cmd += f"--cond-index {int(self.seq_config['cond_idx']/self.seq_config['frame_interval']) * self.seq_config['frame_interval']} "
        cmd += f"--cond-index {self.seq_config['cond_idx']} "
        cmd += f"--SAM3D-index {self.seq_config['cond_idx']} "
        cmd += f"--out-dir {out_dir} "

        if self.vis:
            cmd += f"--vis "        
        print(cmd)
        os.system(cmd)

    def ho3d_SAM3D_post_process(self, scene_name, **kwargs):
        self.print_header(f"Copy SAM3D results for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        src_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_pts/{id}/"
        dst_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_post_process/{id}/"

        if self.vis:
            cmd = f"cd {home_dir}/Documents/project/vggt && "
            cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python SAM3D_post_process_vis.py "
            cmd += f"--out-dir {dst_dir} "       
            print(cmd)
            os.system(cmd)
            return
                   
        if self.rebuild:
            cmd = f"rm -rf {dst_dir}/"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/vggt && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python SAM3D_post_process.py "
        cmd += f"--src-dir {src_dir} "
        cmd += f"--dst-dir {dst_dir} "       
        print(cmd)
        os.system(cmd)

    def align_mesh_image(self, scene_name, **kwargs):
        self.print_header(f"align mesh and image for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        data_dir = f"{self.dataset_dir}/{scene_name}/3D_gen/{id}/"
        out_dir = f"{self.dataset_dir}/{scene_name}/align_mesh_image/{id}/"

        if self.vis:
            cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-2.1/ && "
            cmd += f"{self.conda_dir}/envs/hunyuan_2.1/bin/python "
            cmd += f"rerun_vis.py --data_dir {out_dir}"
            print(cmd)
            os.system(cmd)
            return

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-2.1/ && "
        cmd += f"{self.conda_dir}/envs/hunyuan_2.1/bin/python align_mesh_image_dr.py "
        cmd += f"--mesh_path {data_dir}/white_mesh_remesh.obj "
        cmd += f"--image_path {data_dir}/input_image_prepared_0000.png "
        cmd += f"--mask_path {data_dir}/input_mask_prepared_0000.png "
        cmd += f"--output_dir {out_dir}/ "
        print(cmd)
        os.system(cmd)

    def get_pts_observed(self, scene_name, **kwargs):
        self.print_header(f"get pts observed for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{self.dataset_dir}/{scene_name}/pts_observed"

        if self.vis:
            cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-2.1/ && "
            cmd += f"{self.conda_dir}/envs/hunyuan_2.1/bin/python "
            cmd += f"rerun_vis.py --data_dir {out_dir}"
            print(cmd)
            os.system(cmd)
            return

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-2.1/ && "
        cmd += f"{self.conda_dir}/envs/hunyuan_2.1/bin/python get_pts_observed.py "
        cmd += f"--image_file {data_dir}/rgb/{id}.jpg "
        cmd += f"--mask_file {data_dir}/mask_obj/{id}.png "
        cmd += f"--depth_file {data_dir}/depth_fs/{id}.png "
        cmd += f"--meta_file {data_dir}/meta/0000.pkl "
        cmd += f"--output_dir {out_dir}/ "
        print(cmd)
        os.system(cmd)

    def align_corres(self, scene_name, **kwargs):
        self.print_header(f"align corres for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        cond_data_dir = f"{self.dataset_dir}/{scene_name}/align_mesh_image/{id}/"
        query_data_dir = f"{self.dataset_dir}/{scene_name}/pts_observed/"
        out_dir = f"{self.dataset_dir}/{scene_name}/features/"

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-2.1/ && "
        cmd += f"{self.conda_dir}/envs/threestudio/bin/python align_corres.py "
        cmd += f"--cond_img_f {cond_data_dir}/image.png "
        cmd += f"--cond_depth_f {cond_data_dir}/depth.png "
        cmd += f"--cond_camera_f {cond_data_dir}/camera.json "
        cmd += f"--query_img_f {query_data_dir}/image_filtered.png "
        cmd += f"--query_depth_f {query_data_dir}/depth_filtered.png "
        cmd += f"--query_camera_f {query_data_dir}/camera.json "
        cmd += f"--in_dir {cond_data_dir} "
        cmd += f"--out_dir {out_dir} "
        cmd += f"--mute"
        print(cmd)
        os.system(cmd)

    def align_pcs(self, scene_name, **kwargs):
        self.print_header(f"align pcs for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        data_dir = f"{self.dataset_dir}/{scene_name}/"
        out_dir = f"{data_dir}/align/"
        
        if self.vis:
            cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-2.1/ && "
            cmd += f"{self.conda_dir}/envs/threestudio/bin/python align_vis.py "
            cmd += f"--data_dir {data_dir} "
            cmd += f"--id {id} "
            print(cmd)
            os.system(cmd)
            return
        
        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-2.1/ && "
        cmd += f"{self.conda_dir}/envs/threestudio/bin/python align_pcs.py "
        cmd += f"--query_index {id} "
        cmd += f"--data_dir {data_dir}/features/ "
        cmd += f"--cond_camera_f {data_dir}/align_mesh_image/{id}/camera.json "
        cmd += f"--cond_img_f {data_dir}/align_mesh_image/{id}/image.png "
        cmd += f"--out_dir {out_dir} "
        cmd += f"--mute "
        print(cmd)
        os.system(cmd)

    def scale_3D_gen(self, scene_name, **kwargs):
        self.print_header(f"scale 3D gen for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"

        mesh_file = f"{self.dataset_dir}/{scene_name}/3D_gen/{id}/white_mesh_remesh.obj"
        align_file = f"{self.dataset_dir}/{scene_name}/align/{id}.json"

        cmd = f"cd {self.code_dir}/code && "
        cmd += f"{self.conda_dir}/envs/threestudio/bin/python scale_3D_gen.py "
        cmd += f"--mesh_file {mesh_file} "
        cmd += f"--align_file {align_file} "
        print(cmd)
        os.system(cmd)
    

    def mesh2SDF(self, scene_name, **kwargs):
        self.print_header(f"mesh2SDF for {scene_name}")
        cond_idx = f"{self.seq_config['cond_idx']:04d}"

        data_dir = f"{self.dataset_dir}/{scene_name}/3D_gen/{cond_idx}/"
        out_dir = f"{self.dataset_dir}/{scene_name}/SDF_gen/"

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {self.code_dir}/code && "
        cmd += f"{self.conda_dir}/envs/threestudio/bin/python mesh2SDF.py "
        cmd += f"--mesh_path {data_dir}/white_mesh_remesh.obj "
        cmd += f"--output_path {out_dir}/white_mesh_remesh_sdf.ply "

        print(cmd)
        os.system(cmd)

    def hunyuan_omni(self, scene_name, **kwargs):
        self.print_header(f"hunyuan omni for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        data_dir = f"{self.dataset_dir}/{scene_name}/"
        out_dir = f"{data_dir}/hunyuan_omni/"

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-Omni && "
        cmd += f"{self.conda_dir}/envs/omni/bin/python inference.py "
        cmd += f"--image_file {data_dir}/3D_gen/{id}/input_image_prepared_{id}.png "
        cmd += f"--mesh_files {data_dir}/align/query_raw_align.ply "
        cmd += f"--save_dir {out_dir} "
        print(cmd)
        os.system(cmd)

    def get_depth_from_foundation_stereo(self,seq_name, **kwargs):
        #iterate over all scene_name in dataset_dir
        self.print_header(f"get depth from foundation stereo for {seq_name}")
        if self.rebuild:
            cmd = f"cd {self.dataset_dir}/{seq_name} && rm -rf depth_fs ply_fs"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/FoundationStereo && " \
            f"proxychains4 {self.conda_dir}/envs/foundation_stereo/bin/python " \
            "scripts/run_video.py " \
            f"--left_dir {self.dataset_dir}/{seq_name}/ir/ " \
            f"--right_dir {self.dataset_dir}/{seq_name}/ir/ " \
            f"--intrinsic_file {self.dataset_dir}/{seq_name}/meta/0000.pkl " \
            f"--ckpt_dir ./pretrained_models/model_best_bp2.pth " \
            f"--out_dir {self.dataset_dir}/{seq_name}/depth_fs/ " \
            f"--ply_dir {self.dataset_dir}/{seq_name}/ply_fs/ " \
            f"--ply_interval 10"            
            # --realsense \
            # --denoise_cloud \
            # --visualize_cloud \
        print(cmd)
        os.system(cmd)

    def convert_depth_to_ply(self, scene_name, **kwargs):
        #iterate over all scene_name in dataset_dir
        self.print_header(f"convert depth to ply for {scene_name}")
        if self.rebuild:
            cmd = f"cd {self.dataset_dir}/{scene_name} && rm -rf ply_fs"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/MonoGS && " \
            f"{self.conda_dir}/envs/py38cu116_GS_ICP/bin/python " \
            f"realsense_get_ply.py --input_dir {self.dataset_dir}/{scene_name}"
        print(cmd)
        os.system(cmd)

    def ZED_parse_data(self,scene_name, **kwargs):
        self.print_header(f"ZED parse data for {scene_name}")

        cmd = f"cd {home_dir}/Documents/project/zed-sdk/recording/export/svo/python && " \
            f"{self.conda_dir}/bin/python svo_export.py --mode 2" \
            f" --input_svo_file {self.dataset_dir}/{scene_name}/{scene_name}.svo2" \
            f" --output_path_dir {self.dataset_dir}/{scene_name}/"
        os.system(cmd)

    def ZED_read_data(self):
        self.print_header(f"ZED read data")
        cmd = "cd /usr/local/zed/tools && " \
              "./ZED_Explorer "
        os.system(cmd)

    def realsense_read_data(self):
        self.print_header(f"Realsense read data")
        cmd = f"cd {home_dir}/Documents/project/MonoGS && " \
            f"{self.conda_dir}/envs/py38cu116_GS_ICP/bin/python " \
            "realsense.py --config configs/live/realsense_rgbd.yaml"
        os.system(cmd)

    def print_header(self, process):
        header = f"========== start: {process} =========="
        print("-"*len(header))
        print(header)
        print("-"*len(header))

    def hot3d_sync_to_local(self, scene_name, **kwargs):
        self.print_header(f"hot3D sync to local for {scene_name}")
        remote_path = f"shibo@10.30.47.2:/data1/shibo/Documents/dataset/HOT3D/{scene_name}"
        local_path = f"{home_dir}/Documents/dataset_local/HOT3D"
        cmd = f"cd {local_path} && rsync -azvp {remote_path} ."
        print(cmd)
        os.system(cmd)

    def hot3d_sync_hands_to_local(self, scene_name, **kwargs):
        self.print_header(f"hot3D sync hands prediction to local for {scene_name}")
        remote_path = f"shibo@10.30.47.2:/data1/shibo/Documents/dataset/BundleSDF/HO3D_v3/train/{scene_name}/hands"
        local_path = f"{home_dir}/Documents/dataset/BundleSDF/HO3D_v3/train/{scene_name}/"
        cmd = f"rsync -azvp {remote_path} {local_path}"
        print(cmd)
        os.system(cmd)        

    def hot3d_cp_images(self, scene_name, **kwargs):
        self.print_header(f"hot3D copy images for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/processed/undistorted/"
        output_dir = f"{self.dataset_dir}/{scene_name}/processed/images/"

        if self.rebuild:
            cmd = f"rm -rf {output_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {self.code_dir}/scripts && "
        cmd += f"python cp_images.py --data_dir {data_dir} --out_dir {output_dir} "
        cmd += f"--frame_interval {self.seq_config['frame_interval']}"
        print(cmd)
        os.system(cmd)

    def hot3d_gen_meta(self, scene_name, **kwargs):
        self.print_header(f"hot3d generate intrinsics for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/processed/undistorted/"
        out_dir = f"{self.dataset_dir}/{scene_name}/processed/meta/"

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {self.code_dir}/scripts && "
        cmd += f"python gen_meta.py --data_dir {data_dir} --out_dir {out_dir} "
        cmd += f"--frame_interval {self.seq_config['frame_interval']}"
        print(cmd)
        os.system(cmd)
        
    
    def hot3d_get_undistorted_stereo(self, scene_name, **kwargs):
        self.print_header(f"hot3d get undistorted stereo for {scene_name}")
        out_dir = f"{self.dataset_dir}/{scene_name}/processed/undistorted/"

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/hot3d/hot3d && " \
              f"python viewer.py --sequence_folder {self.dataset_dir}/{scene_name} " \
              f"--object_library_folder {self.dataset_dir}/assets/ " \
              f"--mano_model_folder {self.dataset_dir}/body_models/ " \
              f"--hand_type MANO " \
              f"--out_dir {out_dir} " \
              f"--interval 1 " \
              f"--headless "
        
        print(cmd)
        os.system(cmd)

    def hot3d_get_pose_in_cam(self, scene_name, **kwargs):
        self.print_header(f"hot3d get pose in cam for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/processed/"
        out_dir = f"{self.dataset_dir}/{scene_name}/processed/"

        if self.rebuild:
            cmd = f"cd {home_dir}/Documents/project/hot3d/hot3d && "
            cmd += f"rm -rf {out_dir}/undistorted/hand_poses_in_cam && "
            cmd += f"rm -rf {out_dir}/undistorted/object_poses_in_cam "
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/hot3d/hot3d && "
        cmd += f"python obj2camera_hand2camer.py "\
               f"--data_dir {data_dir}/undistorted " \
               f"--out_dir {out_dir}/undistorted " \
               f"--interval 1 " \
               f"--headless "
        print(cmd)
        os.system(cmd)

    def hot3d_validate_pose_in_cam(self, scene_name, **kwargs):
        self.print_header(f"hot3d validate pose in cam for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/processed/"
        out_dir = f"{self.dataset_dir}/{scene_name}/processed/"

        if self.rebuild:
            cmd = f"cd {home_dir}/Documents/project/hot3d/hot3d && "
            cmd += f"rm -rf {out_dir}/undistorted/project_in_cam "
            print(cmd)
            os.system(cmd)


        cmd = f"cd {home_dir}/Documents/project/hot3d/hot3d && "
        cmd += f"python project_validate.py "
        cmd += f"--data_dir {data_dir}/undistorted " \
               f"--out_dir {out_dir}/undistorted " \
               f"--sample_vertices 100 "
        print(cmd)
        os.system(cmd)

    def zed_sync_data_to_local(self, scene_name, **kwargs):
        self.print_header(f"vggt sync data to local for {scene_name}")

        frame_interval = self.seq_config['frame_interval']

        data_dir = f"{home_dir}/Documents/dataset/WonderHOI/ZED/{scene_name}/"
        output_dir = f"{vggt_code_dir}/examples_ZED/{scene_name}/"

        if self.rebuild:
            cmd = f"rm -rf {output_dir}/*"
            print(cmd)
            os.system(cmd)

        os.makedirs(output_dir, exist_ok=True)

        cmd = f"cd {vggt_code_dir} && python cp_meta.py --data_path={data_dir}/meta --output_dir={output_dir}/meta_origin --frame_interval={frame_interval}"
        print(cmd)
        os.system(cmd)

        cmd = f"cd {vggt_code_dir} && python cp_origin.py --data_path={data_dir}/mask_obj --output_dir={output_dir}/mask_obj_origin --frame_interval={frame_interval}"
        print(cmd)
        os.system(cmd)

        cmd = f"cd {vggt_code_dir} && python crop_image.py --image_dir={output_dir}/mask_obj_origin --output_dir={output_dir}/images --meta_path={output_dir}/meta_origin/0000.pkl"
        print(cmd)
        os.system(cmd)

        cmd = f"cd {vggt_code_dir} && python cp_origin.py --data_path={data_dir}/depth_fs --output_dir={output_dir}/depth_fs_origin --frame_interval={frame_interval}"
        print(cmd)
        os.system(cmd)

        cmd = f"cd {vggt_code_dir} && python crop_image.py --image_dir={output_dir}/depth_fs_origin --output_dir={output_dir}/depth_fs --meta_path={output_dir}/meta_origin/0000.pkl"
        print(cmd)
        os.system(cmd)

        cmd = f"cp -rf {data_dir}/3D_gen {output_dir} && cp -rf {data_dir}/align_mesh_image {output_dir}"
        print(cmd)
        os.system(cmd)


    def zed_joint_optimization(self, scene_name, **kwargs):
        self.print_header(f"vggt joint optimization for {scene_name}")
        
        data_dir = f"{vggt_code_dir}/examples_ZED/{scene_name}"
        out_dir = f"{vggt_code_dir}/output/{scene_name}"

        if self.vis:
            cmd = f"cd {vggt_code_dir}/viewer && "
            cmd += f"{self.conda_dir}/envs/threestudio/bin/python viewer_step.py "
            cmd += f"--result_folder {out_dir}/results/ "
            cmd += f"--vis_only_register "
            print(cmd)
            os.system(cmd)
            return

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python demo_colmap.py "
        cmd += f"--scene_dir {data_dir} "
        cmd += f"--max_query_pts 200 --query_frame_num 0 --vis_thresh 0.40 --max_reproj_error 3 --shared_camera "
        cmd += f"--output_dir {out_dir} --use_calibrated_intrinsic --max_frame_num 100 --frame_interval 1 --dataset_type ZED"
        print(cmd)
        os.system(cmd)

    def ho3d_keyframe_optimization(self, scene_name, **kwargs):
        self.print_header(f"ho3d keyframe optimization for {scene_name}")
        
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{vggt_code_dir}/output/{scene_name}"
        frame_interval = self.seq_config['frame_interval']
        frame_number = self.seq_config['frame_number']

        if self.vis:
            cmd = f"cd {vggt_code_dir} && "
            cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python viewer/viewer_step.py "
            cmd += f"--result_folder {out_dir}/results/ "
            cmd += f"--vis_only_register "
            cmd += f"--vis_only_keyframes "            
            cmd += f"--num_frames {frame_number} "
            cmd += f"--log_aligned_mesh 0 "
            print(cmd)
            os.system(cmd)
            return

        if self.rebuild:
            cmd = f"cd {out_dir} && rm -rf results track_* pipeline*.log data_processed"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python robust_hoi.py "
        cmd += f"--scene_dir {data_dir} "
        cmd += f"--max_frame_num {frame_number * frame_interval -1} --frame_interval {frame_interval} --dataset_type HO3D "
        cmd += f"--output_dir {out_dir} "
        cmd += f"--cond_index {int(self.seq_config['cond_idx'] / self.seq_config['frame_interval'])} "
        cmd += f"--cond_index_raw {int(self.seq_config['cond_idx'])} "
        print(cmd)
        os.system(cmd)

    def ho3d_align_gen_3d(self, scene_name, **kwargs):
        self.print_header(f"align gen 3D for {scene_name}")
        
        data_dir = f"{vggt_code_dir}/output/{scene_name}/"
        out_dir = f"{vggt_code_dir}/output/{scene_name}/gen_3d_aligned/"
        frame_number = self.seq_config['frame_number']

        if self.vis:
            cmd = f"cd {vggt_code_dir} && "
            cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python viewer/viewer_step.py "
            cmd += f"--result_folder {out_dir}/../results/ "
            cmd += f"--vis_only_register "
            cmd += f"--vis_only_keyframes "            
            cmd += f"--num_frames {frame_number} "
            cmd += f"--log_aligned_mesh 1 "
            print(cmd)
            os.system(cmd)
            return

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python align_gen_3d.py "
        cmd += f"--input_dir {data_dir} "
        cmd += f"--output_dir {out_dir} "
        cmd += f"--init_pose_image_idx {int(self.seq_config['cond_idx'] / self.seq_config['frame_interval'])} "


        print(cmd)
        os.system(cmd)

    def ho3d_align_gen_3d_omni(self, scene_name, **kwargs):
        self.print_header(f"align gen 3D omni for {scene_name}")
        
        keyframe_dir = f"{vggt_code_dir}/output/{scene_name}/results/"
        gen3d_aligned_dir = f"{vggt_code_dir}/output/{scene_name}/gen_3d_aligned/refined/"
        out_dir = f"{vggt_code_dir}/output/{scene_name}/gen_3d_aligned_omni/"
        frame_number = self.seq_config['frame_number']

        if self.vis:
            cmd = f"cd {vggt_code_dir} && "
            cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python viewer/viewer_step.py "
            cmd += f"--result_folder {out_dir}/../results/ "
            cmd += f"--vis_only_register "
            cmd += f"--vis_only_keyframes "            
            cmd += f"--num_frames {frame_number} "
            cmd += f"--log_aligned_mesh 1 "
            print(cmd)
            os.system(cmd)
            return

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python align_gen_3d_omni.py "
        cmd += f"--keyframe_dir {keyframe_dir} "
        cmd += f"--gen3d_aligned_dir {gen3d_aligned_dir} "
        cmd += f"--output_dir {out_dir} "

        print(cmd)
        os.system(cmd)

    def ho3d_obj_sdf_optimization(self, scene_name, **kwargs):
        self.print_header(f"ho3d object sdf optimization for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/"
        result_dir = f"{vggt_code_dir}/output/{scene_name}/results/"
        out_dir = f"{vggt_code_dir}/output/{scene_name}/sdf_optimization/"
        id = f"{self.seq_config['cond_idx']:04d}"

        cmd = "export CC=gcc-11 && export CXX=g++-11 && export CUDAHOSTCXX=g++-11 &&"
        cmd += f"cd {vggt_code_dir}/third_party/instant-nsr-pl && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python launch.py "
        cmd += f"--config configs/neus-mixed.yaml "
        cmd += f"--train " # or --test
        cmd += f"dataset.root_dir={result_dir} "
        cmd += f"dataset.sam3d_root_dir={data_dir}/SAM3D_aligned_post_process/{id}/ "
        cmd += f"--exp_dir {out_dir} "
        # cmd += f"--resume {out_dir}/neus-mixed/@20260204-223258/ckpt/epoch=0-step=10000.ckpt "

        print(cmd)
        os.system(cmd)

        
    def _ho3d_eval(self, scene_name, hand_fit_mode):
        self.print_header(f"vggt ho3d evaluate ({hand_fit_mode}) for {scene_name}")

        result_dir = f"{vggt_code_dir}/output/{scene_name}/results/"
        out_dir = f"{vggt_code_dir}/output/{scene_name}/eval_{hand_fit_mode}/"

        if self.vis:
            cmd = f"cd {vggt_code_dir} && "
            cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python viewer_pose.py "
            cmd += f"--result_folder {result_dir} "
            print(cmd)
            os.system(cmd)
            return

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python eval.py "
        cmd += f"--result_folder {result_dir} "
        cmd += f"--out_dir {out_dir} "
        cmd += f"--hand_fit_mode {hand_fit_mode}"  # choices: intrinsic, trans, rot
        print(cmd)
        os.system(cmd)

    def ho3d_eval_intrinsic(self, scene_name, **kwargs):
        self._ho3d_eval(scene_name, "intrinsic")

    def ho3d_eval_trans(self, scene_name, **kwargs):
        self._ho3d_eval(scene_name, "trans")

    def ho3d_eval_rot(self, scene_name, **kwargs):
        self._ho3d_eval(scene_name, "rot")

    def ho3d_inpaint(self, scene_name, **kwargs):
        self.print_header(f"vggt ho3d inpaint for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{self.dataset_dir}/{scene_name}/inpaint"
        frame_number = self.seq_config['frame_number']
        frame_interval = self.seq_config["frame_interval"]
        cond_select_strategy = self.seq_config["cond_select_strategy"]
        if self.rebuild:
            cmd = f" rm -rf {out_dir}/{self.seq_config['cond_idx']:04d} "
            print(f"{cmd}")
            os.system(cmd)

        cmd = f"python inpaint.py "
        cmd += f"--image_dir {data_dir} "
        cmd += f"--out_dir {out_dir} "
        cmd += f"--cond_select_strategy {cond_select_strategy} "
        cmd += f"--cond_view {self.seq_config['cond_idx']} "
        cmd += f"--max_frame_num {frame_number * frame_interval -1} --frame_interval {frame_interval} "
        print(f"{cmd}")
        os.system(cmd)

    def hoi_pipeline_data_preprocess(self, scene_name, **kwargs):
        self.print_header(f"hoi pipeline data preprocess for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{self.dataset_dir}/{scene_name}/pipeline_preprocess/"
        frame_interval = self.seq_config['frame_interval']
        frame_number = self.seq_config['frame_number']

        if self.rebuild:
            cmd = f"rm -rf {out_dir}"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python robust_hoi_pipeline/pipeline_data_preprocess.py "
        cmd += f"--data_dir {data_dir} "
        cmd += f"--output_dir {out_dir} "
        cmd += f"--start 0 --end {frame_number * frame_interval - 1} --interval {frame_interval} "
        cmd += f"--cond_index {self.seq_config['cond_idx']} "
        print(cmd)
        os.system(cmd)

    def hoi_pipeline_data_preprocess_sam3d_neus(self, scene_name, **kwargs):
        self.print_header(f"hoi pipeline sam3d neus initialization for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        data_dir = f"{self.dataset_dir}/{scene_name}"
        result_dir = f"{vggt_code_dir}/output/{scene_name}/"
        out_dir = f"{self.dataset_dir}/{scene_name}/SAM3D_aligned_post_process/{id}/neus/"

        if self.rebuild:
            cmd = f"rm -rf {out_dir}"
            print(cmd)
            os.system(cmd)
        
        CONDA_PREFIX = f"{self.conda_dir}/envs/vggsfm_tmp"
        cmd = f'''cd {vggt_code_dir} && export PATH={CONDA_PREFIX}/bin:$PATH && export CC={CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc &&  export CXX={CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++ && '''
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python robust_hoi_pipeline/pipeline_neus_init.py "
        cmd += f"--data_dir {data_dir} "
        cmd += f"--result_dir {result_dir} "
        cmd += f"--output_dir {out_dir} "
        cmd += f"--cond_index {self.seq_config['cond_idx']} "
        cmd += f"--robust_hoi_weight 0.0 " # set to 0.0 to disable robust hoi in neus initialization
        cmd += f"--sam3d_weight 1.0 " # only run sam3d neus initialization without robust hoi
        print(cmd)
        os.system(cmd)

    def hoi_pipeline_get_corres(self, scene_name, **kwargs):
        self.print_header(f"hoi pipeline get correspondences for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}/pipeline_preprocess"
        out_dir = f"{self.dataset_dir}/{scene_name}/pipeline_corres"

        if self.rebuild:
            cmd = f"rm -rf {out_dir}"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python3 robust_hoi_pipeline/pipeline_get_corres.py "
        cmd += f"--data_dir {data_dir} "
        cmd += f"--out_dir {out_dir} "
        cmd += f"--cond_index {self.seq_config['cond_idx']} "
        cmd += f"--matching_backend vggsfm " # vggsfm or mast3r
        print(cmd)
        os.system(cmd)

    def hoi_pipeline_neus_init(self, scene_name, **kwargs):
        self.print_header(f"hoi pipeline neus initialization for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{vggt_code_dir}/output/{scene_name}/pipeline_neus_init"

        if self.rebuild:
            cmd = f"rm -rf {out_dir}"
            print(cmd)
            os.system(cmd)
        
        CONDA_PREFIX = f"{self.conda_dir}/envs/vggsfm_tmp"
        cmd = f'''cd {vggt_code_dir} && export PATH={CONDA_PREFIX}/bin:$PATH && export CC={CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc &&  export CXX={CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++ && '''
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python robust_hoi_pipeline/pipeline_neus_init.py "
        cmd += f"--data_dir {data_dir} "
        cmd += f"--output_dir {out_dir} "
        cmd += f"--cond_index {self.seq_config['cond_idx']} "
        print(cmd)
        os.system(cmd)

    def hoi_pipeline_joint_opt(self, scene_name, **kwargs):
        self.print_header(f"hoi pipeline joint optimization for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{vggt_code_dir}/output/{scene_name}"
        id = f"{self.seq_config['cond_idx']:04d}"

        if self.vis:
            cmd = f"cd {vggt_code_dir} && "
            cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python robust_hoi_pipeline/pipeline_joint_opt_vis.py "
            cmd += f"--data_dir {data_dir} "
            cmd += f"--output_dir {out_dir} "
            cmd += f"--cond_index {self.seq_config['cond_idx']} "
            print(cmd)
            os.system(cmd)
            return
        
        if self.eval:
            cmd = f"cd {vggt_code_dir} && "
            cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python robust_hoi_pipeline/pipeline_joint_opt_eval.py "
            cmd += f"--result_folder {out_dir}/pipeline_joint_opt/ "
            cmd += f"--out_dir {out_dir}/pipeline_joint_opt/eval/ "
            cmd += f"--SAM3D_dir {data_dir}/SAM3D_aligned_post_process "
            cmd += f"--cond_index {self.seq_config['cond_idx']} "

            print(cmd)
            os.system(cmd)
            return

        if self.rebuild:
            cmd = f"rm -rf {out_dir}/pipeline_joint_opt"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python robust_hoi_pipeline/pipeline_joint_opt.py "
        cmd += f"--data_dir {data_dir} "
        cmd += f"--output_dir {out_dir} "
        cmd += f"--cond_index {self.seq_config['cond_idx']} "
        print(cmd)
        os.system(cmd)

    def hoi_pipeline_reg_remaining(self, scene_name, **kwargs):
        self.print_header(f"hoi pipeline register remaining for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{vggt_code_dir}/output/{scene_name}"

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python robust_hoi_pipeline/pipeline_reg_remaining.py "
        cmd += f"--data_dir {data_dir} "
        cmd += f"--output_dir {out_dir} "
        cmd += f"--cond_index {self.seq_config['cond_idx']} "
        print(cmd)
        os.system(cmd)

    def _eval_sum(self, scene_name, fit_mode):
        """Run evaluation for a specific stage (before/after pose refinement)"""
        self.print_header(f"evaluate summary ({fit_mode}) for {scene_name}")

        output_file = f"{vggt_code_dir}/output/metrics_summary/eval{fit_mode}.txt"

        if self.rebuild:
            cmd = f"rm -rf {output_file}"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python extract_jsons.py"
        cmd += f" --parent_dir output "
        cmd += f" --metric_folder eval{fit_mode} "
        cmd += f" --output_file {output_file} "
        print(cmd)
        os.system(cmd)

    def hoi_pipeline_HY_gen(self, scene_name, **kwargs):
        self.print_header(f"Generate object 3D model from Hunyuan for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        out_dir = f"{vggt_code_dir}/output/{scene_name}/pipeline_hunyuan_3d/"
        data_dir = f"{self.dataset_dir}/{scene_name}"
                   
        if self.rebuild:
            cmd = f"rm -rf {out_dir}/*"
            print(cmd)
            os.system(cmd)

        # python example.py --image_file $IMAGE_FILE --output_dir $scene_output_dir
        cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-2.1/ && "
        cmd += f"{self.conda_dir}/envs/hunyuan_2.1/bin/python example.py "
        # cmd += f"--image_file {self.dataset_dir}/{scene_name}/mask_obj/{id}.png "
        cmd += f"--image_file {data_dir}/SAM3D/{id}/rendered_novel_view_yup.png "
        cmd += f"--output_dir {out_dir}"
        print(cmd)
        os.system(cmd)

    def hoi_pipeline_HY_omni_gen(self, scene_name, **kwargs):
        self.print_header(f"hunyuan omni for {scene_name}")
        id = f"{self.seq_config['cond_idx']:04d}"
        data_dir = f"{self.dataset_dir}/{scene_name}/"
        out_dir = f"{vggt_code_dir}/output/{scene_name}/"
        save_dir = f"{out_dir}/pipeline_hunyuan_omni/"

        if self.rebuild:
            cmd = f"rm -rf {save_dir}/*"
            print(cmd)
            os.system(cmd)

        cmd = f"cd {home_dir}/Documents/project/Hunyuan3D-Omni && "
        cmd += f"{self.conda_dir}/envs/hunyuan_2.1_omni/bin/python inference.py "
        cmd += f"--image_file {data_dir}/SAM3D/{id}/rendered_novel_view_yup.png "
        cmd += f"--mesh_files {out_dir}/pipeline_3D_points_align_with_HY/points_aligned.ply "
        cmd += f"--save_dir {save_dir} "
        print(cmd)
        os.system(cmd)            

    def hoi_pipeline_align_SAM3D_with_HY(self, scene_name, **kwargs):
        self.print_header(f"Align SAM3D with Hunyuan for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{vggt_code_dir}/output/{scene_name}"

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python robust_hoi_pipeline/pipeline_algin_SAM3D_with_HY.py "
        cmd += f"--data_dir {data_dir} "
        cmd += f"--output_dir {out_dir} "
        cmd += f"--cond_index {self.seq_config['cond_idx']} "
        print(cmd)
        os.system(cmd)

    def hoi_pipeline_3D_points_align_with_HY(self, scene_name, **kwargs):
        self.print_header(f"Align 3D track points with Hunyuan for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{vggt_code_dir}/output/{scene_name}"

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python robust_hoi_pipeline/pipeline_3D_points_align_with_HY.py "
        cmd += f"--data_dir {data_dir} "
        cmd += f"--output_dir {out_dir} "
        print(cmd)
        os.system(cmd)

    def hoi_pipeline_HY_to_SAM3D(self, scene_name, **kwargs):
        self.print_header(f"Transform Hunyuan output to SAM3D for {scene_name}")
        data_dir = f"{self.dataset_dir}/{scene_name}"
        out_dir = f"{vggt_code_dir}/output/{scene_name}"

        cmd = f"cd {vggt_code_dir} && "
        cmd += f"{self.conda_dir}/envs/vggsfm_tmp/bin/python robust_hoi_pipeline/pipeline_HY_to_SAM3D.py "
        cmd += f"--data_dir {data_dir} "
        cmd += f"--output_dir {out_dir} "
        print(cmd)
        os.system(cmd)

        

    def eval_sum_intrinsic(self, scene_name, **kwargs):
        self._eval_sum(scene_name, "_intrinsic")

    def eval_sum_trans(self, scene_name, **kwargs):
        self._eval_sum(scene_name, "_trans")

    def eval_sum_rot(self, scene_name, **kwargs):
        self._eval_sum(scene_name, "_rot")

    def eval_sum(self, scene_name, **kwargs):
        self._eval_sum(scene_name, "")        

def main(args, extras):
    # Convert extras list to dictionary
    extras_dict = {}
    for i in range(0, len(extras), 2):
        if i + 1 < len(extras):
            key = extras[i].lstrip('-')  # Remove leading dashes
            value = extras[i + 1]
            # Try to convert value to int or float if possible
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            extras_dict[key] = value
    
    run_wonder_hoi(args, extras_dict).run()

if __name__ == "__main__":
    all_sequences = [
        # ###### ZED dataset ##########
        # "air_gun",
        # "clamp",
        # "cooking_shovel",
        # "cube",
        # "cup1",
        # "cup2",
        # "duck",
        # "fire_fighting_car",
        # "glass_cup",
        # "hammer",
        # "jep_car",
        # "lufei",
        # "mouse",
        # "pitch",
        # "plane",
        # "scisors",
        # "scisors_1",
        # "spoon",
        # "sprayer",
        # "wrench",
        # ###### ZED dataset ##########
        # "air_gun",
        # "clamp",
        # "cooking_shovel",
        # "cube",
        # "cup1",
        # "cup2",
        # "duck",
        # "fire_fighting_car",
        # "glass_cup",
        # "hammer",
        # "jep_car",
        # "lufei",
        # "mouse",
        # "pitch",
        # "plane",
        # "scisors",
        # "scisors_1",
        # "spoon",
        # "sprayer",
        # "wrench",
        # "bottle1",
        # "bottle2",
        # "drug_box",
        # "trans_bottle1" 
        
        

        ###### hot3d ##########
        # "P0003_c701bd11",

        ###### ho3d part 1 ##########
        "ABF12",
        "ABF14",
        "GPMF12",
        "GPMF14",    
        "MC1",
        "MC4",
        "MDF12",
        "MDF14",
        "ShSu10",
        "ShSu14",
        "SM2",
        "SM4",
        "SMu1",
        "SMu40",
        "BB12",
        "BB13", 
        "GSF12",
        "GSF13",
              

        # ###### ho3d full ##########
        # "ABF10",
        # "ABF11",
        # "ABF12",
        # "ABF13",
        # "ABF14",
        # "BB10",
        # "BB11",
        # "BB12",
        # "BB13",
        # "BB14",     
        # "GPMF10",
        # "GPMF11",
        # "GPMF12",
        # "GPMF13",
        # "GPMF14",   
        # "GSF10",
        # "GSF11",
        # "GSF12",
        # "GSF13",
        # "GSF14",
        # "MC1",
        # "MC2",
        # "MC4",
        # "MC5",
        # "MC6",
        # "MDF10",
        # "MDF11",
        # "MDF12",
        # "MDF13",
        # "MDF14",     
        # "ND2",  
        # "SB10",
        # "SB12",
        # "SB14",
        # "ShSu10",
        # "ShSu12",
        # "ShSu13",
        # "ShSu14",
        # "SiBF10",
        # "SiBF11",
        # "SiBF12",
        # "SiBF13",
        # "SiBF14",
        # "SiS1",
        # "SM2",
        # "SM3",
        # "SM4",
        # "SM5",
        # "SMu1",
        # "SMu40",
        # "SMu41",
        # "SMu42",
        # "SS1",
        # "SS2",
        # "SS3",        
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument('--seq_list',
        # choices=all_sequences + ['all'],
        help="Specify the sequence list. Use 'all' to select all sequences.",
        nargs='+',  # To accept multiple values in a list
        required=True  # This makes the argument mandatory
    )

    parser.add_argument('--execute_list', 
        choices=[
                "data_read",
                "data_convert",
                "obj_process",
                "hand_pose_preprocess",
                "hand_pose_postprocess",           
                ], 
        help="Specify the execution option.", 
        nargs='+',  # To accept multiple values in a list
        required=False  # This makes the argument mandatory
    )
    parser.add_argument('--process_list', 
        choices=["realsense_read_data", 
                "realsense_convert_data", 
                "ZED_read_data",
                "ZED_parse_data",
                "convert_depth_to_ply",
                "get_depth_from_foundation_stereo",
                "estimate_hand_pose",
                "estimate_obj_pose",
                "obj_3D_gen",
                "get_pts_observed",
                "align_pcs",
                "scale_3D_gen",
                "fit_hand_intrinsic",
                "align_corres",
                "fit_hand_trans",
                "fit_hand_rot",
                "fit_hand_pose",
                "fit_hand_all",
                "fit_hand_viewer",
                "align_mesh_image",
                "mesh2SDF",
                "hunyuan_omni",
                "hot3d_sync_to_local",
                "hot3d_sync_hands_to_local",
                "hot3d_cp_images",
                "hot3d_gen_meta",
                "hot3d_get_undistorted_stereo",
                "hot3d_get_pose_in_cam",
                "hot3d_validate_pose_in_cam",
                "hot3d_get_depth",
                "hot3d_image_to_video",
                "hot3d_get_mask",
                "hot3d_mask_only_object",
                "hot3d_convet_to_stereo",
                "hot3d_obj_pose",
                "hot3d_estimate_hand_pose",
                "hot3d_interpolate_hamer",
                "zed_joint_optimization",
                "zed_sync_data_to_local",
                "ho3d_obj_3D_gen",
                "ho3d_condition_id",
                "ho3d_obj_SAM3D_gen",
                "ho3d_obj_SAM3D_post_opt_GS",
                "ho3d_obj_SAM3D_post_optimization",
                "ho3d_align_SAM3D_mask",
                "ho3d_align_SAM3D_pts",
                "ho3d_SAM3D_post_process",
                "ho3d_keyframe_optimization",
                "ho3d_align_gen_3d",
                "ho3d_align_gen_3d_omni",
                "ho3d_obj_sdf_optimization",
                "ho3d_estimate_hand_pose",
                "ho3d_interpolate_hamer",
                "ho3d_get_hand_mask",
                "ho3d_get_obj_mask",
                "ho3d_inpaint",
                "hoi_pipeline_neus_init",
                "hoi_pipeline_data_preprocess",
                "hoi_pipeline_data_preprocess_sam3d_neus",
                "hoi_pipeline_get_corres",
                "hoi_pipeline_align_SAM3D_with_HY",
                "hoi_pipeline_3D_points_align_with_HY",
                "hoi_pipeline_HY_to_SAM3D",
                "hoi_pipeline_HY_omni_gen",
                "hoi_pipeline_joint_opt",
                "hoi_pipeline_reg_remaining",
                "hoi_pipeline_HY_gen",
                "ho3d_eval_intrinsic",
                "ho3d_eval_trans",
                "ho3d_eval_rot",
                "eval_sum_intrinsic",
                "eval_sum_trans",
                "eval_sum_rot",
                "eval_sum"
                ],
        help="Specify the process option.", 
        nargs='+',  # To accept multiple values in a list
        required=True  # This makes the argument mandatory
    )
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the process')
    parser.add_argument('--vis', action='store_true', help='Visualize the process')
    parser.add_argument('--eval', action='store_true', help='Evaluate the process')
    parser.add_argument('--dataset_dir', type=str, default=f"{home_dir}/Documents/dataset/BundleSDF/HO3D_v3/train/", help='Dataset directory')
    parser.add_argument('--reconstruction_dir', type=str, default=f"{home_dir}/Documents/project/WonderHOI/code/output_backup/[115][472254d][disable_global_ba]", help='Reconstruction folder')
    parser.add_argument('--conda_type', type=str, default=None, help='Conda environment type: anaconda3 or miniconda3')

    args, extras = parser.parse_known_args()
    
    if 'all' in args.seq_list:
        args.seq_list = all_sequences

    main(args, extras)
