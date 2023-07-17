import json
import math
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import colmap_read
import utils


def prepare_kp_map(img_coords, image):
    im_map = np.zeros((image.shape[0], image.shape[1]))
    # for y, x, _ in img_coords:
    #     im_map[x, y] = 1
    # im_map2 = np.zeros((image.shape[0], image.shape[1]))
    # xs = [du[1] for du in img_coords]
    # ys = [du[0] for du in img_coords]
    # im_map2[xs, ys] = 1
    # indices = [[du[1], du[0]] for du in img_coords]
    # indices = np.array(indices)
    # im_map[indices[:, 0], indices[:, 1]] = 1
    indices = np.array(img_coords)[:, :2]
    im_map[indices[:, 1], indices[:, 0]] = 1
    # indices[:, [0, 1]] = indices[:, [1, 0]]
    return im_map, indices


class CamLocDataset(Dataset):
    """
    Camera localization dataset.
    Access to image, calibration and ground truth data given a dataset directory.
    """

    def __init__(
        self,
        dataset_name="redkitchen",
        images_full_dir="../7_scenes",
        images_dir=None,
        sfm_model_dir="../7scenes_reference_models",
        using_all_images=False,
        training=False,
        augment=False,
        debug=False,
        nb_images=-1,
        nb_preloaded_images=1000,
        aug_rotation=30,
        aug_scale_min=2 / 3,
        aug_scale_max=3 / 2,
        aug_contrast=0.1,
        aug_brightness=0.1,
        image_height=480,
        img_names=None,
        using_filter=True,
        disable_coord_map=True,
        return_true_pose=False,
        whole_image=True,
    ):
        self.sfm_model_dir = Path(f"{sfm_model_dir}/{dataset_name}/sfm_gt")
        if images_dir is None:
            self.images_dir = f"{images_full_dir}/{dataset_name}/images"
        else:
            self.images_dir = images_dir
        self.test_file = self.sfm_model_dir / "list_test.txt"
        self.working_directory = Path(f"data/{dataset_name}")
        self.working_directory.mkdir(parents=True, exist_ok=True)

        self.train = training
        self.recon_images = colmap_read.read_images_binary(f"{self.sfm_model_dir}/images.bin")
        self.recon_cameras = colmap_read.read_cameras_binary(f"{self.sfm_model_dir}/cameras.bin")
        if len(self.recon_cameras) > 1:
            print("Warning, there are more than one camera from the pGT.")
        self.debug = debug
        self.image_height = image_height
        self.nb_images = nb_images  # how many images to use
        self.nb_preloaded_images = nb_preloaded_images  # how many images to preload
        self.using_all_images = using_all_images
        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_contrast = aug_contrast
        self.aug_brightness = aug_brightness
        self.disable_coord_map = disable_coord_map
        self.return_true_pose = return_true_pose
        self.whole_image = whole_image
        self.ds_name = dataset_name

        self.image_transform_pt = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.image_height),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.4
                    ],  # statistics calculated over 7scenes training set, should generalize fairly well
                    std=[0.25],
                ),
            ]
        )
        self.image_shape = None
        self.images_cache = {}
        self.image_name2id = {}
        self.image_id2kp = {}
        self.test_images_list = utils.read_image_list(self.test_file)
        self.img_names = []
        self.invalid_number = 2**64 - 1
        if img_names is None:
            self.read_images()
        else:
            self.img_names = img_names
            self.read_images()

        self.using_keypoints = False
        self.using_filter = using_filter
        self.image_id2kp_sfm = None
        self.bad_points = set([])
        self.good_pid_list = []
        self.image_id2kp = self.read_points_information()

    def return_bounds(self, computed=True):
        if not computed:
            points = colmap_read.read_points3D_binary(
                f"{self.sfm_model_dir}/points3D.bin"
            )
            xyz_arr = np.zeros((len(points), 3), dtype=np.float32)
            for idx, pid in enumerate(points):
                point = points[pid]
                xyz_arr[idx] = point.xyz

            import open3d as o3d

            point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr))
            point_cloud, ind = point_cloud.remove_radius_outlier(
                nb_points=10, radius=0.1
            )
            max_bound = point_cloud.get_max_bound()
            min_bound = point_cloud.get_min_bound()

            # vis = o3d.visualization.Visualizer()
            # vis.create_window(width=1920, height=1025)
            # vis.add_geometry(point_cloud)
            # vis.run()
            # vis.destroy_window()
        else:
            afile = open("data/bounds.pkl", "rb")
            bounds = pickle.load(afile)
            afile.close()
            min_bound, max_bound = bounds[self.ds_name]

        return min_bound, max_bound

    def return_image_ids(self):
        image_ids = []
        for k in self.image_name2id:
            v = self.image_name2id[k]
            image_ids.append(v)
        return image_ids

    def return_cam_mat(self):
        k, cam = list(self.recon_cameras.items())[0]
        cam_mat = torch.eye(3)
        cam_mat[0, 0] = cam.params[0]
        cam_mat[1, 1] = cam.params[0]
        cam_mat[0, 2] = cam.params[1]
        cam_mat[1, 2] = cam.params[2]
        return cam_mat

    def read_images(self):
        for image_id, image in self.recon_images.items():
            self.image_name2id[image.name] = image_id
        if self.using_all_images and len(self.img_names) == 0:
            self.img_names = [name for name in self.image_name2id]
        else:
            if len(self.img_names) == 0:
                if self.train:
                    self.img_names = [
                        name
                        for name in self.image_name2id
                        if name not in self.test_images_list
                    ]
                else:
                    self.img_names = self.test_images_list[:]

        if self.nb_images > 0:
            self.img_names = self.img_names[: self.nb_images]

        # save data of 1000 images into cache to save time
        print(f"Reading images from {self.images_dir}")
        image = io.imread(f"{self.images_dir}/{self.img_names[0]}")
        self.image_shape = image.shape
        for img_name in self.img_names[: self.nb_preloaded_images]:
            image = io.imread(f"{self.images_dir}/{img_name}")
            self.images_cache[img_name] = image

    def read_points_information(self):
        image_id2kp = {}
        for img_name in self.img_names:
            img_id = self.image_name2id[img_name]
            pid_arr = self.recon_images[img_id].point3D_ids
            mask = pid_arr >= 0
            pid_arr = pid_arr[mask]
            xy_arr = self.recon_images[img_id].xys[mask].astype(int)
            img_coords = []
            for idx, pid in enumerate(pid_arr):
                if pid == self.invalid_number:
                    continue
                if not self.using_filter or pid in self.good_pid_list:
                    x, y = xy_arr[idx]
                    img_coords.append([x, y, pid])
            image_id2kp[str(img_id)] = img_coords
        if self.using_keypoints:
            self.image_id2kp_sfm = {}
            for img_id in tqdm(image_id2kp, desc="Processing SFM keypoints"):
                new_dict = {}
                data1 = image_id2kp[str(img_id)]
                for x, y, pid in data1:
                    new_dict[(x, y)] = pid
                self.image_id2kp_sfm[img_id] = new_dict
        return image_id2kp

    def prepare_coord_map(self, img_id, image):
        img_coords = self.image_id2kp[str(img_id)]
        recon_points = colmap_read.read_points3D_binary(f"{self.sfm_model_dir}/points3D.bin")
        assert len(img_coords) > 0
        if self.disable_coord_map:
            return None, img_coords, None, None, None
        coord_map = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        pid_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32) - 1
        xyz_arr = []
        uv_arr = []
        for x, y, pid in img_coords:
            point3d = recon_points[pid]
            coord_map[y, x] = point3d.xyz
            pid_map[y, x] = pid
            xyz_arr.append(point3d.xyz)
            uv_arr.append([y, x])
        return coord_map, img_coords, pid_map, np.array(xyz_arr), np.array(uv_arr)

    def prepare_pid2uv(self, img_id):
        pid2uv = {}
        data = self.image_id2kp[str(img_id)]
        for u, v, pid in data:
            pid2uv[pid] = [u, v]
        return pid2uv

    def prepare_pid2uv_all(self):
        pid2uv = {}
        for img_id in self.image_id2kp:
            data = self.image_id2kp[img_id]
            for u, v, pid in data:
                if pid not in pid2uv:
                    pid2uv[pid] = [[u, v]]
                else:
                    pid2uv[pid].append([u, v])
        return pid2uv

    def prepare_cam_info(self, img_id):
        camera_id = self.recon_images[img_id].camera_id
        focal_length = self.recon_cameras[camera_id].params[0]
        qvec = self.recon_images[img_id].qvec
        tvec = self.recon_images[img_id].tvec
        if not self.return_true_pose:
            pose = utils.return_pose_mat_no_inv(qvec, tvec)
        else:
            pose = utils.return_pose_mat(qvec, tvec)
        return camera_id, focal_length, pose, qvec, tvec

    def prepare_cam_info_no_inverse(self, img_id):
        camera_id = self.recon_images[img_id].camera_id
        focal_length = self.recon_cameras[camera_id].params[0]
        qvec = self.recon_images[img_id].qvec
        tvec = self.recon_images[img_id].tvec
        pose_no_inv = utils.return_pose_mat_no_inv(qvec, tvec)
        cam_mat = np.eye(3)
        cam_mat[0, 0] = focal_length
        cam_mat[1, 1] = focal_length
        cam_mat[0, 2] = self.image_shape[1] / 2
        cam_mat[1, 2] = self.image_shape[0] / 2
        return pose_no_inv, cam_mat

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        out_dict = {}
        img_name = self.img_names[idx]
        img_id = self.image_name2id[img_name]

        if img_name in self.images_cache:
            image = self.images_cache[img_name]
        else:
            image = io.imread(f"{self.images_dir}/{img_name}")

        coord_map, img_coords, pid_map, xyz_arr, uv_arr = self.prepare_coord_map(
            img_id, image
        )
        kp_map_original, kp_indices = prepare_kp_map(img_coords, image)
        out_dict.update({"kp_map": kp_map_original})

        if coord_map is not None:
            out_dict.update(
                {
                    "coord_map": coord_map,
                    "pid_map": pid_map,
                }
            )

        out_dict.update({"image_ori": image})

        camera_id, focal_length, pose, qvec, tvec = self.prepare_cam_info(img_id)

        camera_params = self.recon_cameras[camera_id].params
        out_dict.update({"params": camera_params})

        # image will be normalized to standard height, adjust focal length as well
        f_scale_factor = self.image_height / image.shape[0]
        focal_length *= f_scale_factor

        if self.augment:
            scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
            angle = random.uniform(-self.aug_rotation, self.aug_rotation)
            out_dict.update({"scale_factor": scale_factor, "angle": angle})

            # augment input image
            cur_image_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(int(self.image_height * scale_factor)),
                    transforms.ColorJitter(
                        brightness=self.aug_brightness, contrast=self.aug_contrast
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4], std=[0.25]),
                ]
            )

            image_transformed = cur_image_transform(image)

            if not self.whole_image:
                keypoints, valid_keypoints, kp_map, mask = utils.transform_kp_aug_fast(
                    kp_indices,
                    self.image_height,
                    scale_factor,
                    image,
                    image_transformed,
                    angle,
                )

                if coord_map is not None:
                    old_keypoints = kp_indices[:, 0], kp_indices[:, 1]
                    old_coord_map = out_dict["coord_map"]
                    coord_map2 = np.zeros(
                        [image_transformed.shape[1], image_transformed.shape[2], 3],
                        dtype=np.float32,
                    )
                    coord_map2[
                        valid_keypoints[:, 0], valid_keypoints[:, 1]
                    ] = old_coord_map[old_keypoints[1][mask], old_keypoints[0][mask]]

                    out_dict.update(
                        {
                            "kp_map": kp_map,
                            "coord_map": coord_map2,
                            "coord_map_c": np.rollaxis(coord_map2, 2, 0),
                        }
                    )
                else:
                    out_dict.update({"kp_map": kp_map, "kp_aug": valid_keypoints})

            focal_length *= scale_factor
            if angle != 0:
                image = utils.rotate_image(
                    image_transformed, angle, 1, "constant", cval=-1
                )

                # rotate ground truth camera pose
                pose = torch.tensor(pose).float()
                angle = angle * math.pi / 180
                pose_rot = torch.eye(4)
                pose_rot[0, 0] = math.cos(angle)
                pose_rot[0, 1] = -math.sin(angle)
                pose_rot[1, 0] = math.sin(angle)
                pose_rot[1, 1] = math.cos(angle)
                pose = torch.matmul(pose_rot.inverse(), pose)

            else:
                image = image_transformed
        else:
            image = self.image_transform_pt(image)
            out_dict.update({"qvec": qvec, "tvec": tvec})

        out_dict.update({"pose": pose, "focal": focal_length})
        out_dict.update(
            {"image": image, "image_name": self.img_names[idx], "image_id": img_id}
        )
        return out_dict


if __name__ == "__main__":
    ds2bounds = {}
    for ds in ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]:
        train_ds = CamLocDataset(
            training=True,
            debug=True,
            dataset_name=ds,
            using_all_images=False,
            nb_images=10,
            nb_preloaded_images=2000,
            augment=True,
            disable_coord_map=False,
            using_filter=False,
            return_true_pose=False,
            whole_image=False,
        )
        b1, b2 = train_ds.return_bounds(computed=True)
        # ds2bounds[ds] = [b1.tolist(), b2.tolist()]
    # ddir = "data/bounds.pkl"
    # with open(ddir, "wb") as handle:
    #     pickle.dump(ds2bounds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print()
