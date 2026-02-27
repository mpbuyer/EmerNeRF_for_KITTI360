# datasets/kitti360.py
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from datasets.base.pixel_source import ScenePixelSource
from datasets.base.scene_dataset import SceneDataset
from datasets.base.split_wrapper import SplitWrapper

logger = logging.getLogger()


def _sorted_glob_png(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
    return sorted(files)


def _frame_id_from_path(p: str) -> int:
    # KITTI-360 frames are zero-padded like 0000000000.png
    stem = os.path.splitext(os.path.basename(p))[0]
    return int(stem)


def _load_kitti360_prect(calib_perspective_txt: str, cam_id: int = 0) -> np.ndarray:
    """
    Parse calibration/perspective.txt and return P_rect_{cam_id:02d} as (3,4).
    KITTI-360 uses 'P_rect_00:' lines in perspective.txt.
    """
    assert os.path.exists(calib_perspective_txt), f"Missing: {calib_perspective_txt}"
    key = f"P_rect_{cam_id:02d}:"
    with open(calib_perspective_txt, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(key):
                vals = [float(x) for x in line[len(key) :].strip().split()]
                if len(vals) != 12:
                    raise ValueError(f"Expected 12 floats for {key} in {calib_perspective_txt}")
                return np.array(vals, dtype=np.float64).reshape(3, 4)
    raise KeyError(f"Could not find '{key}' in {calib_perspective_txt}")


def _load_cam0_to_world(cam0_to_world_txt: str) -> Dict[int, np.ndarray]:
    """
    Parse data_poses/.../cam0_to_world.txt into {frame_id: T_cam0_world(4x4)}.
    Known format: first column is frame id, followed by 12 floats for a 3x4 matrix.
    """
    assert os.path.exists(cam0_to_world_txt), f"Missing: {cam0_to_world_txt}"
    out: Dict[int, np.ndarray] = {}
    with open(cam0_to_world_txt, "r") as f:
        for row in f:
            line = row.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 13:
                continue
            frame_id = int(parts[0])
            mat = np.array([float(x) for x in parts[1:13]], dtype=np.float64).reshape(3, 4)
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = mat
            out[frame_id] = T
    return out


class KITTI360PixelSource(ScenePixelSource):
    """
    Minimal KITTI-360 pixel source for monocular perspective camera (image_00).
    Intended for a single camera (num_cams=1). Assumes OpenCV-style camera axes
    (x right, y down, z forward) for KITTI-360 perspective cameras. :contentReference[oaicite:1]{index=1}
    """

    # KITTI-360 perspective cameras follow OpenCV axis convention -> identity.
    OPENCV2DATASET = np.eye(4, dtype=np.float64)

    def __init__(
        self,
        pixel_data_config: OmegaConf,
        data_root: str,
        sequence_name: str,  # e.g. "2013_05_28_drive_0007_sync"
        cam_id: int = 0,  # 0 -> image_00 (left perspective)
        start_timestep: int = 0,
        end_timestep: int = -1,
        device: torch.device = torch.device("cpu"),
    ):
        # KITTI-360 does not provide EmerNeRF-format dynamic/sky masks by default.
        if getattr(pixel_data_config, "load_dynamic_mask", False):
            pixel_data_config.load_dynamic_mask = False
            logger.info("[Pixel][KITTI360] Overriding load_dynamic_mask=False (no masks by default).")
        if getattr(pixel_data_config, "load_sky_mask", False):
            pixel_data_config.load_sky_mask = False
            logger.info("[Pixel][KITTI360] Overriding load_sky_mask=False (no masks by default).")

        super().__init__(pixel_data_config, device=device)

        if self.num_cams != 1:
            raise NotImplementedError("KITTI360PixelSource currently supports num_cams=1 only.")

        self.data_root = data_root
        self.sequence_name = sequence_name
        self.cam_id = cam_id
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep

        self.seq_dir_2d = os.path.join(self.data_root, "data_2d_raw", self.sequence_name)
        self.seq_dir_poses = os.path.join(self.data_root, "data_poses", self.sequence_name)

        self.image_dir = os.path.join(self.seq_dir_2d, f"image_{cam_id:02d}", "data_rect")
        self.calib_perspective = os.path.join(self.data_root, "calibration", "perspective.txt")
        self.cam0_to_world_txt = os.path.join(self.seq_dir_poses, "cam0_to_world.txt")

        self._poses_by_frame = _load_cam0_to_world(self.cam0_to_world_txt)
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self) -> None:
        img_paths = _sorted_glob_png(self.image_dir)
        if not img_paths:
            raise FileNotFoundError(f"No PNGs found in {self.image_dir}")

        # Keep frames that have an entry in cam0_to_world.txt.
        # cam0_to_world uses a 1-based frame index in the first column per KITTI360 scripts/issues. :contentReference[oaicite:2]{index=2}
        # Image filenames are 0-based. We align by using (frame_id_from_png + 1) as pose key.
        keep: List[Tuple[str, int, int]] = []
        for p in img_paths:
            img_frame0 = _frame_id_from_path(p)  # 0-based
            pose_frame1 = img_frame0 + 1         # 1-based
            if pose_frame1 in self._poses_by_frame:
                keep.append((p, img_frame0, pose_frame1))

        if not keep:
            raise RuntimeError(
                "No frames had both images and cam0_to_world poses. "
                "Check your KITTI-360 download and folder structure."
            )

        # Apply start/end timesteps on the synchronized image index space.
        # Here timestep == image index in the filtered list.
        n = len(keep)
        start = max(0, int(self.start_timestep))
        if self.end_timestep == -1:
            end = n
        else:
            end = min(n, int(self.end_timestep) + 1)  # inclusive -> exclusive
        if end <= start:
            raise ValueError(f"Invalid timestep range: start={start}, end={end}, available={n}")

        keep = keep[start:end]
        self.start_timestep = start
        self.end_timestep = end

        self.img_filepaths = np.array([k[0] for k in keep])
        self._img_frame0 = np.array([k[1] for k in keep], dtype=np.int64)
        self._pose_frame1 = np.array([k[2] for k in keep], dtype=np.int64)

        # Feature path convention is EmerNeRF-specific; keep consistent with other datasets.
        self.feat_filepaths = np.array(
            [
                p.replace("data_2d_raw", "data_2d_raw")
                .replace("/data_rect/", f"/data_{self.data_cfg.feature_model_type}/")
                .replace(".png", ".npy")
                for p in self.img_filepaths
            ]
        )

        # No masks by default.
        self.dynamic_mask_filepaths = np.array([])
        self.sky_mask_filepaths = np.array([])

    def load_calibrations(self) -> None:
        # Get original image size from first frame (avoid hardcoding).
        with Image.open(self.img_filepaths[0]) as im:
            W0, H0 = im.size  # PIL: (W, H)
        self.ORIGINAL_SIZE = [[H0, W0]]  # (H, W) like other sources

        # P_rect_00 contains K in the left 3x3 block.
        P = _load_kitti360_prect(self.calib_perspective, cam_id=self.cam_id)
        K = P[:, :3].copy()

        # Scale intrinsics to configured load_size.
        Ht, Wt = int(self.data_cfg.load_size[0]), int(self.data_cfg.load_size[1])
        sx = Wt / float(W0)
        sy = Ht / float(H0)
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

        # Build per-image intrinsics, c2w, cam_ids, timestamps.
        intrinsics = []
        cam_to_worlds = []
        cam_ids = []
        timestamps = []
        timesteps = []

        # Make the first pose the origin (like Waymo/NuScenes implementations).
        T0 = self._poses_by_frame[int(self._pose_frame1[0])]
        T0_inv = np.linalg.inv(T0)

        for local_t, pose_frame1 in enumerate(self._pose_frame1.tolist()):
            Tw = T0_inv @ self._poses_by_frame[int(pose_frame1)]
            # KITTI-360 perspective camera axes are already OpenCV-like, so OPENCV2DATASET is identity.
            Tw = Tw @ self.OPENCV2DATASET

            cam_to_worlds.append(Tw)
            intrinsics.append(K)
            cam_ids.append(0)  # single camera

            # Use local index as timestamp/timestep (simple, works with EmerNeRF’s normalization).
            timestamps.append(float(local_t))
            timesteps.append(int(local_t))

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        self.cam_ids = torch.from_numpy(np.array(cam_ids, dtype=np.int64)).long()

        # ScenePixelSource expects these underscore fields.
        self._timestamps = torch.from_numpy(np.array(timestamps, dtype=np.float32)).float()
        self._timesteps = torch.from_numpy(np.array(timesteps, dtype=np.int64)).long()


class KITTI360Dataset(SceneDataset):
    dataset: str = "kitti360"

    def __init__(self, data_cfg: OmegaConf) -> None:
        super().__init__(data_cfg)
        assert self.data_cfg.dataset == "kitti360"

        # Required config fields:
        # - data_root: path to KITTI-360 root containing calibration/, data_2d_raw/, data_poses/
        # - sequence_name: "2013_05_28_drive_0007_sync"
        # - pixel_source.num_cams must be 1 for this minimal loader
        self.data_root = self.data_cfg.data_root
        self.sequence_name = self.data_cfg.sequence_name
        self.cam_id = int(getattr(self.data_cfg, "cam_id", 0))  # default left perspective

        # Build sources (pixel only by default).
        self.pixel_source, self.lidar_source = self.build_data_source()

        # aabb can fall back to pixel source if lidar is None (SceneDataset.get_aabb()).
        self.aabb = self.get_aabb()

        (
            self.train_timesteps,
            self.test_timesteps,
            self.train_indices,
            self.test_indices,
        ) = self.split_train_test()

        pixel_sets, lidar_sets = self.build_split_wrapper()
        self.train_pixel_set, self.test_pixel_set, self.full_pixel_set = pixel_sets
        self.train_lidar_set, self.test_lidar_set, self.full_lidar_set = lidar_sets

    def build_data_source(self):
        pixel_source = None
        lidar_source = None
        all_timestamps = []

        load_pixel = (
            self.data_cfg.pixel_source.load_rgb
            or self.data_cfg.pixel_source.load_feature
            or getattr(self.data_cfg.pixel_source, "load_sky_mask", False)
            or getattr(self.data_cfg.pixel_source, "load_dynamic_mask", False)
        )
        if load_pixel:
            pixel_source = KITTI360PixelSource(
                pixel_data_config=self.data_cfg.pixel_source,
                data_root=self.data_root,
                sequence_name=self.sequence_name,
                cam_id=self.cam_id,
                start_timestep=self.data_cfg.start_timestep,
                end_timestep=self.data_cfg.end_timestep,
                device=self.device,
            )
            pixel_source.to(self.device)
            all_timestamps.append(pixel_source.timestamps)

        # Minimal version: no LiDAR support (enable later if needed).
        # If you want to extend: implement KITTI360LiDARSource similar to NuScenesLiDARSource/WaymoLiDARSource.

        assert len(all_timestamps) > 0, "No data source is loaded (pixel_source.load_* all false?)."

        all_timestamps = torch.cat(all_timestamps, dim=0)
        all_timestamps = (all_timestamps - all_timestamps.min()) / (all_timestamps.max() - all_timestamps.min() + 1e-8)
        all_timestamps = all_timestamps.float()

        if pixel_source is not None:
            pixel_source.register_normalized_timestamps(all_timestamps[: len(pixel_source.timestamps)])

        return pixel_source, lidar_source

    def build_split_wrapper(self):
        train_pixel_set = test_pixel_set = full_pixel_set = None
        train_lidar_set = test_lidar_set = full_lidar_set = None

        if self.pixel_source is not None:
            train_pixel_set = SplitWrapper(
                datasource=self.pixel_source,
                split_indices=self.train_indices,
                split="train",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )
            full_pixel_set = SplitWrapper(
                datasource=self.pixel_source,
                split_indices=np.arange(self.pixel_source.num_imgs).tolist(),
                split="full",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )
            if len(self.test_indices) > 0:
                test_pixel_set = SplitWrapper(
                    datasource=self.pixel_source,
                    split_indices=self.test_indices,
                    split="test",
                    ray_batch_size=self.data_cfg.ray_batch_size,
                )

        return (train_pixel_set, test_pixel_set, full_pixel_set), (train_lidar_set, test_lidar_set, full_lidar_set)

    def split_train_test(self):
        # Use the same convention as WaymoDataset. :contentReference[oaicite:3]{index=3}
        if self.data_cfg.pixel_source.test_image_stride != 0:
            test_timesteps = np.arange(
                self.data_cfg.pixel_source.test_image_stride,
                self.num_img_timesteps,
                self.data_cfg.pixel_source.test_image_stride,
            )
        else:
            test_timesteps = np.array([], dtype=np.int64)

        train_timesteps = np.array([i for i in range(self.num_img_timesteps) if i not in set(test_timesteps)], dtype=np.int64)

        train_indices: List[int] = []
        test_indices: List[int] = []
        for t in range(self.num_img_timesteps):
            # num_cams == 1, but keep the general pattern used elsewhere.
            if t in set(train_timesteps.tolist()):
                for cam in range(self.pixel_source.num_cams):
                    train_indices.append(t * self.pixel_source.num_cams + cam)
            elif t in set(test_timesteps.tolist()):
                for cam in range(self.pixel_source.num_cams):
                    test_indices.append(t * self.pixel_source.num_cams + cam)

        return train_timesteps, test_timesteps, train_indices, test_indices
    def save_videos(self, video_dict: dict, **kwargs):
        # Same pattern as WaymoDataset.save_videos()
        return save_videos(
            render_results=video_dict,
            save_pth=kwargs["save_pth"],
            num_timestamps=kwargs["num_timestamps"],
            keys=kwargs["keys"],
            num_cams=kwargs["num_cams"],
            fps=kwargs["fps"],
            verbose=kwargs["verbose"],
            save_seperate_video=kwargs["save_seperate_video"],
        )

    def render_data_videos(
        self,
        save_pth: str,
        split: str = "full",
        fps: int = 24,
        verbose: bool = True,
    ):
        """
        Render a quick 'data.mp4' preview video for KITTI-360 monocular.
        For num_cams=1, the number of frames in the video equals num_img_timesteps
        (i.e., the number of images loaded after start/end_timestep trimming).
        Mirrors WaymoDataset.render_data_videos() but omits LiDAR overlays. :contentReference[oaicite:2]{index=2}
        """
        if self.pixel_source is None:
            raise RuntimeError("pixel_source is None; enable data.pixel_source.load_rgb=True")

        # pick the right split wrapper (same control flow as Waymo)
        if split == "full":
            pixel_dataset = self.full_pixel_set
        elif split == "train":
            pixel_dataset = self.train_pixel_set
        elif split == "test":
            pixel_dataset = self.test_pixel_set
        else:
            raise NotImplementedError(f"Split {split} not supported")

        rgb_imgs = []
        dynamic_objects = []
        sky_masks = []
        feature_pca_colors = []

        for i in trange(len(pixel_dataset), desc="Rendering data videos", dynamic_ncols=True):
            data_dict = pixel_dataset[i]

            if "pixels" in data_dict:
                rgb_imgs.append(data_dict["pixels"].cpu().numpy())

            if "dynamic_masks" in data_dict:
                dynamic_objects.append(
                    (data_dict["dynamic_masks"].unsqueeze(-1) * data_dict["pixels"]).cpu().numpy()
                )

            if "sky_masks" in data_dict:
                sky_masks.append(data_dict["sky_masks"].cpu().numpy())

            if "features" in data_dict:
                # visualize features via the registered PCA parameters (same idea as Waymo)
                feats = data_dict["features"]
                feats = feats @ self.pixel_source.feat_dimension_reduction_mat
                feats = (feats - self.pixel_source.feat_color_min) / (
                    self.pixel_source.feat_color_max - self.pixel_source.feat_color_min
                )
                feats = feats.clamp(0, 1)
                feature_pca_colors.append(feats.cpu().numpy())

        video_dict = {
            "gt_rgbs": rgb_imgs,
            "gt_feature_pca_colors": feature_pca_colors,
            # optional:
            # "gt_dynamic_objects": dynamic_objects,
            # "gt_sky_masks": sky_masks,
        }
        video_dict = {k: v for k, v in video_dict.items() if len(v) > 0}

        return self.save_videos(
            video_dict,
            save_pth=save_pth,
            num_timestamps=self.num_img_timesteps,
            keys=video_dict.keys(),
            num_cams=self.pixel_source.num_cams,
            fps=fps,
            verbose=verbose,
            save_seperate_video=False,
        )


