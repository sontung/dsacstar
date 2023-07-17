import json
import math
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import skimage
import torch
from pykdtree.kdtree import KDTree
from scipy.spatial import KDTree as scipy_KDTree
from scipy.spatial.transform import Rotation as Rotation
from skimage.transform import rotate
from tqdm import tqdm

from network_dsac import DsacNet

WARNING = False


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def create_sampling_matrix(out_w, out_h):
    mat = np.zeros((out_h, out_w, 2))
    for x in range(out_w):
        for y in range(out_h):
            mat[y, x] = [x * 8 + 4, y * 8 + 4]
    return mat


def read_gt_slam_poses(file_):
    file_hd = open(file_, "r")
    qvec = {}
    tvec = {}
    focal = {}
    lines = file_hd.readlines()
    for line in lines:
        line = line[:-1]
        name, qw, qx, qy, qz, tx, ty, tz, f = line.split(" ")
        qw, qx, qy, qz, tx, ty, tz, f = map(float, [qw, qx, qy, qz, tx, ty, tz, f])
        qvec[name] = [qw, qx, qy, qz]
        tvec[name] = [tx, ty, tz]
        focal[name] = f
    return qvec, tvec, focal


def check_coordinates_quality(desc):
    vec = torch.sum(torch.abs(desc), dim=-1)
    num_ = torch.nonzero(vec).shape[0]
    return num_ / desc.shape[0]


def check_coordinates_duplicability(desc_list):
    res = []
    for desc in desc_list:
        desc = desc.detach().cpu().numpy()
        cond = np.sum(np.abs(desc), 1)
        desc = desc[np.nonzero(cond)]
        tree = KDTree(desc)
        dist, indices = tree.query(desc, 2)
        good = np.sum(dist[:, 1] > 0)
        res.append(good / dist.shape[0])
    return res


# @profile
def check_points_quality(points, coord_map):
    acc_arr = []
    assert len(points.shape) == 2
    acc = 0
    for x, y in points:
        if torch.sum(torch.abs(coord_map[y, x])) > 0:
            acc += 1
    acc_arr.append(acc / points.shape[0])
    return acc_arr


# @profile
def select_coordinates(image_coordinates, coord_map, trees):
    """
    return 3d coordinates for each image coordinate,
    if there is no matched 3d coordinates, query the closest pixel from the tree
    """
    sampled_coord_map = np.zeros((image_coordinates.shape[0], 3))
    coord_map = coord_map.cpu().numpy()
    bad_indices = []

    for j, (x, y) in enumerate(image_coordinates):
        xyz = coord_map[y, x]

        if np.sum(np.abs(xyz)) == 0:
            tree, tree2, coord_mat = trees
            dist, ind = tree2.query([y, x])
            if dist < 5:
                x2, y2 = coord_mat[ind]
                xyz = coord_map[x2, y2]
                sampled_coord_map[j] = xyz
            else:
                bad_indices.append(j)
        else:
            sampled_coord_map[j] = xyz
    return sampled_coord_map, bad_indices


def select_coordinates_fast(image_coordinates, coord_map, trees, dist_=10):
    """
    faster version of select_coordinates
    """
    # find zero coordinates
    coord_map = coord_map.cpu().numpy()
    selected_coordinates = coord_map[image_coordinates[:, 1], image_coordinates[:, 0]]

    # query closest coordinates for zero coordinates
    bad_image_coordinates = np.copy(image_coordinates)
    bad_image_coordinates[:, [1, 0]] = bad_image_coordinates[:, [0, 1]]
    tree, coord_mat = trees
    dist_arr, ind_arr = tree.query(bad_image_coordinates)

    # replace zero coordinates
    nearby_condition = dist_arr < dist_
    condition_arr2 = np.nonzero(nearby_condition)[0]
    ind_arr = ind_arr[condition_arr2]
    bad_indices = np.nonzero(1 - nearby_condition)[0]
    better_points = coord_mat[ind_arr]
    selected_coordinates[condition_arr2] = coord_map[
        better_points[:, 0], better_points[:, 1]
    ]

    return selected_coordinates, bad_indices, dist_arr


def divide_into_cubes(min_bound, max_bound, cube_size=0.5):
    steps = (max_bound - min_bound) / cube_size
    steps_x, steps_y, steps_z = np.ceil(steps).astype(int)
    xyz = np.zeros((steps_x * steps_y * steps_z, 6), dtype=np.float64)
    arr = np.ones_like(xyz) * cube_size
    idx = 0
    for x in range(steps_x):
        for y in range(steps_y):
            for z in range(steps_z):
                xyz[idx, :3] = np.array([x, y, z])
                xyz[idx, 3:6] = np.array([x + 1, y + 1, z + 1])
                idx += 1
    cubes = arr * xyz + np.hstack([min_bound, min_bound])
    return cubes


def kp_map_to_tree(kp_maps):
    """
    convert a keypoint map into a tree to query closest point
    """
    trees = []
    kp_maps = kp_maps.numpy()
    for i in range(kp_maps.shape[0]):
        kp_map = kp_maps[i]
        coord_mat = np.transpose(np.nonzero(kp_map))
        tree = KDTree(coord_mat)
        trees.append([tree, coord_mat])
    return trees


def compute_error(pose_gt, pose):
    est_pose = np.vstack([pose.Rt, [0, 0, 0, 1]])
    est_pose = np.linalg.inv(est_pose)

    error = compute_error_max_rot_trans(pose_gt, est_pose)
    return error


def compute_error_max_rot_trans(pgt_pose, est_pose):
    """
    Compute the pose error.
    Expects poses to map camera coordinate to world coordinates.
    """

    # calculate pose errors
    t_err = float(np.linalg.norm(pgt_pose[0:3, 3] - est_pose[0:3, 3]))

    r_err = est_pose[0:3, 0:3] @ np.transpose(pgt_pose[0:3, 0:3])
    r_err = cv2.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi

    return max(r_err, t_err * 100), r_err, t_err * 100


def convert_pose_data(pose_data):
    """
    Expects path to file with one pose per line.
    Input pose is expected to map world to camera coordinates.
    Output pose maps camera to world coordinates.
    Pose format: file qw qx qy qz tx ty tz (f)
    Return dictionary that maps a file name to a tuple of (4x4 pose, focal_length)
    Sets focal_length to None if not contained in file.
    """

    # create a dict from the poses with file name as key
    pose_dict = {}
    for pose_string in pose_data:

        pose_string = pose_string.split()
        file_name = pose_string[0]

        pose_q = np.array(pose_string[1:5])
        pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
        pose_t = np.array(pose_string[5:8])
        pose_R = Rotation.from_quat(pose_q).as_matrix()

        pose_4x4 = np.identity(4)
        pose_4x4[0:3, 0:3] = pose_R
        pose_4x4[0:3, 3] = pose_t

        # convert world->cam to cam->world for evaluation
        print(pose_4x4)
        pose_4x4 = np.linalg.inv(pose_4x4)

        if len(pose_string) > 8:
            focal_length = float(pose_string[8])
        else:
            focal_length = None

        pose_dict[file_name] = (pose_4x4, focal_length)

    return pose_dict


def read_pose_data(file_name):
    """
    Expects path to file with one pose per line.
    Input pose is expected to map world to camera coordinates.
    Output pose maps camera to world coordinates.
    Pose format: file qw qx qy qz tx ty tz (f)
    Return dictionary that maps a file name to a tuple of (4x4 pose, focal_length)
    Sets focal_length to None if not contained in file.
    """

    with open(file_name, "r") as f:
        pose_data = f.readlines()

    # create a dict from the poses with file name as key
    pose_dict = {}
    for pose_string in pose_data:

        pose_string = pose_string.split()
        file_name = pose_string[0]

        pose_q = np.array(pose_string[1:5])
        pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
        pose_t = np.array(pose_string[5:8])
        pose_R = Rotation.from_quat(pose_q).as_matrix()

        pose_4x4 = np.identity(4)
        pose_4x4[0:3, 0:3] = pose_R
        pose_4x4[0:3, 3] = pose_t

        # convert world->cam to cam->world for evaluation
        pose_4x4 = np.linalg.inv(pose_4x4)

        if len(pose_string) > 8:
            focal_length = float(pose_string[8])
        else:
            focal_length = None

        pose_dict[file_name] = (pose_4x4, focal_length, pose_string)

    return pose_dict


def return_pose_mat(pose_q, pose_t):
    pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
    pose_R = Rotation.from_quat(pose_q).as_matrix()

    pose_4x4 = np.identity(4)
    pose_4x4[0:3, 0:3] = pose_R
    pose_4x4[0:3, 3] = pose_t

    # convert world->cam to cam->world for evaluation
    pose_4x4_inv = np.linalg.inv(pose_4x4)
    return pose_4x4_inv


def return_pose_mat_no_inv(pose_q, pose_t):
    pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
    pose_R = Rotation.from_quat(pose_q).as_matrix()

    pose_4x4 = np.identity(4)
    pose_4x4[0:3, 0:3] = pose_R
    pose_4x4[0:3, 3] = pose_t

    return pose_4x4


def read_points3D(in_dir="sfm_models/points3D.txt"):
    """
    pid => pid, xyz, rgb
    """
    sys.stdin = open(in_dir, "r")
    lines = sys.stdin.readlines()
    data = {}
    for line in lines:
        if line[0] == "#":
            continue
        numbers = line[:-1].split(" ")
        numbers = list(map(float, numbers))
        point3d_id, x, y, z, r, g, b = numbers[:7]
        tracks = list(map(int, numbers[8:]))
        point3d_id = int(point3d_id)
        data[point3d_id] = tracks
    return data


def read_image_list(in_dir):
    sys.stdin = open(in_dir, "r")
    lines = sys.stdin.readlines()
    data = []
    for line in lines:
        data.append(line[:-1])
    return data


def read_points3D_coordinates(in_dir="sfm_models/points3D.txt", return_mat=False):
    """
    mapper from pid to xyz rgb
    """
    sys.stdin = open(in_dir, "r")
    lines = sys.stdin.readlines()
    data = {}
    for line in lines:
        if line[0] == "#":
            continue
        numbers = line[:-1].split(" ")[:7]
        numbers = list(map(float, numbers))
        point3d_id, x, y, z, r, g, b = numbers[:7]
        point3d_id = int(point3d_id)
        data[point3d_id] = [x, y, z, r, g, b]
    if return_mat:
        coord_mat = np.zeros((len(data), 3))
        color_mat = np.zeros((len(data), 3))
        id_mat = np.zeros((len(data),))
        for idx, pid in enumerate(data.keys()):
            x, y, z, r, g, b = data[pid]
            id_mat[idx] = pid
            coord_mat[idx] = [x, y, z]
            color_mat[idx] = [r, g, b]
        return data, coord_mat, color_mat, id_mat
    return data


def build_co_visibility_graph(image2pose):
    """
    co-visibility matrix between database images
    image_id_to_visibilities: image_id -> image_id2 -> number of co-visible points
    image_id_to_top_k: image_id -> list of co-visible images (sorted by the number of co-visible points)
    """
    pid2image_id = {}
    image_id_to_visibilities = {}
    for image_id in image2pose:
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose[image_id]
        image_id_to_visibilities[image_id] = {}
        for x, y, p3d_id in points2d_meaningful:
            if p3d_id > 0:
                if p3d_id not in pid2image_id:
                    pid2image_id[p3d_id] = [image_id]
                else:
                    pid2image_id[p3d_id].append(image_id)

    for pid in pid2image_id:
        images = pid2image_id[pid]
        for image_id in images:
            for image_id2 in images:
                if image_id2 != image_id:
                    if image_id2 not in image_id_to_visibilities[image_id]:
                        image_id_to_visibilities[image_id][image_id2] = 1
                    else:
                        image_id_to_visibilities[image_id][image_id2] += 1

    image_id_to_top_k = {}
    for image_id in image_id_to_visibilities:
        visibilities = image_id_to_visibilities[image_id]
        images = list(visibilities.keys())
        images = sorted(images, key=lambda du: visibilities[du], reverse=True)
        image_id_to_top_k[image_id] = images
    return image_id_to_visibilities, image_id_to_top_k


def read_cameras(cam_dir="sfm_ws_hblab/cameras.txt"):
    sys.stdin = open(cam_dir, "r")
    lines = sys.stdin.readlines()
    data = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line[0] == "#":
            idx += 1
            continue
        else:
            line = line[:-1].split(" ")
            cam_id, model, width, height = line[:4]
            cam_id, width, height = map(int, [cam_id, width, height])
            params = list(map(float, line[4:]))
            data[cam_id] = [model, width, height, params]
            idx += 1
    return data


def read_name2id(image2pose):
    name2id = {}
    for img_id in image2pose:
        img_name = image2pose[img_id][0]
        name2id[img_name] = img_id
    return name2id


def read_images(in_dir="sfm_models/images.txt", by_im_name=False):
    """
    this returns a dict:
    data[image_id] = [image_name, points2d_meaningful, cam_pose, cam_id]
    """
    try:
        sys.stdin = open(in_dir, "r")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find images.txt inside {'/'.join(str(in_dir).split('/')[:-1])}"
        )
    lines = sys.stdin.readlines()
    data = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line[0] == "#":
            idx += 1
            continue
        else:
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, image_name = line[:-1].split(
                " "
            )
            cam_pose = list(map(float, [qw, qx, qy, qz, tx, ty, tz]))
            image_id, cam_id = list(map(int, [image_id, cam_id]))
            points2d = list(map(float, lines[idx + 1][:-1].split(" ")))
            points2d_meaningful = []  # [x, y, point 3d id]
            for i in range(0, len(points2d), 3):
                point = (points2d[i], points2d[i + 1], int(points2d[i + 2]))
                points2d_meaningful.append(point)
            if not by_im_name:
                data[image_id] = [image_name, points2d_meaningful, cam_pose, cam_id]
            else:
                data[image_name] = cam_pose
            idx += 2
    return data


def read_pid2images(image2pose):
    """
    maps point 3d id to 2d features that see this point.
    """
    data = {}
    for image_id in image2pose:
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose[image_id]
        for x, y, p3d_id in points2d_meaningful:
            if p3d_id > 0:
                if p3d_id not in data:
                    data[p3d_id] = [(image_id, image_name, x, y)]
                else:
                    data[p3d_id].append((image_id, image_name, x, y))
    return data


def filter_points3d(
    recon_model,
    images_full_dir="/home/n11373598/work/redkitchen/images",
    d2_file="data/redkitchen/d2_keypoints_db.pkl",
    result_dir=Path("filter.pkl"),
):
    if result_dir.is_file():
        file = open(result_dir, "rb")
        filter_ = pickle.load(file)
        file.close()
        return filter_

    tracks = []
    errors = []
    points_data = recon_model.points3D
    images_data = recon_model.images
    for pid in points_data:
        tracks.append(points_data[pid].track.length())
        errors.append(points_data[pid].error)
    file = open(d2_file, "rb")
    invalid_number = 2**64 - 1
    d2_data = pickle.load(file)
    file.close()
    pid2distances = {pid: [] for pid in points_data}
    point_filter = {pid: 0 for pid in points_data}

    for img_id in tqdm(images_data, desc="Filtering 3d points"):
        img_name = images_data[img_id].name
        d2_keypoints, d2_responses = d2_data[f"{images_full_dir}/{img_name}"]
        indices = np.argsort(d2_responses)[-1000:]
        d2_keypoints = d2_keypoints[indices][:, :2]
        tree = KDTree(d2_keypoints)

        points = images_data[img_id].points2D
        point3d_ids = []
        sfm_keypoints = []
        for point in points:
            if point.point3D_id == invalid_number:
                continue
            sfm_keypoints.append(point.xy)
            point3d_ids.append(point.point3D_id)

        # compute the closest distances from sfm keypoints to d2 keypoints
        sfm_keypoints = np.array(sfm_keypoints, dtype=np.float32)
        distance_matrix, indices = tree.query(sfm_keypoints)
        for ind, dist in enumerate(distance_matrix):
            pid2distances[point3d_ids[ind]].append(dist)

    for pid in points_data:
        if len(pid2distances[pid]) == 0:
            del pid2distances[pid]
        else:
            pid2distances[pid] = np.mean(pid2distances[pid])

    for pid in pid2distances:
        if pid2distances[pid] < 5 and points_data[pid].error < 2:
            point_filter[pid] = 1

    points_per_image = []
    for img_id in tqdm(images_data):
        points = images_data[img_id].points2D
        res = 0
        for point in points:
            if point.point3D_id == invalid_number:
                continue
            if point_filter[point.point3D_id] == 1:
                res += 1
        points_per_image.append(res)
    if np.min(points_per_image) < 10:
        print("Warning, some images have low number of keypoints")
        print(
            np.mean(points_per_image),
            np.min(points_per_image),
            np.max(points_per_image),
        )

    with open(result_dir, "wb") as handle:
        pickle.dump(point_filter, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return point_filter


def compute_reproj_error(
    cam_mat, gt_pose, pixel_grid_crop, scene_coords, min_depth=0.1
):
    gt_pose_inv = gt_pose[0].inverse()[0:3, :]
    gt_pose_inv = gt_pose_inv.to(scene_coords.device)  # 3, 4

    # scene coordinates to camera coordinate
    camera_coords = torch.mm(gt_pose_inv, scene_coords)

    # re-project predicted scene coordinates
    reprojection_error = torch.mm(cam_mat, camera_coords)
    reprojection_error[2].clamp_(min=min_depth)  # avoid division by zero
    reprojection_error = reprojection_error[0:2] / reprojection_error[2]
    reprojection_error = reprojection_error - pixel_grid_crop

    return reprojection_error


def find_mean_coordinates(trainset_loader):
    mean = torch.zeros((3,))
    count = 0

    for sample in tqdm(trainset_loader, desc="Finding mean coordinates"):
        # use mean of ground truth scene coordinates
        gt_pose = sample["pose"]
        mean += gt_pose[0, 0:3, 3]
        count += 1

    mean /= count
    return mean


def return_pixel_grid():
    pixel_grid = torch.zeros((2, math.ceil(5000 / 8), math.ceil(5000 / 8)))

    for x in range(0, pixel_grid.size(2)):
        for y in range(0, pixel_grid.size(1)):
            pixel_grid[0, y, x] = x * 8 + 4
            pixel_grid[1, y, x] = y * 8 + 4
    return pixel_grid


def normalize(a, b, arr, using_pt=False):
    if using_pt:
        min_val = torch.min(arr)
        max_val = torch.max(arr)
    else:
        min_val = np.min(arr)
        max_val = np.max(arr)
    arr = (b - a) * (arr - min_val) / (max_val - min_val) + a
    return arr


def check_collision(coords, switch_channel=True):
    if switch_channel:
        coords2 = coords.permute([1, 0]).cpu().numpy()
    else:
        coords2 = coords.cpu().numpy()
    _, indices = np.unique(coords2, axis=0, return_index=True)
    return indices


def return_pixel_grid_dsac():
    pixel_grid = torch.zeros((2, math.ceil(5000 / 8), math.ceil(5000 / 8)))

    for x in range(0, pixel_grid.size(2)):
        for y in range(0, pixel_grid.size(1)):
            pixel_grid[0, y, x] = x * 8 + 4
            pixel_grid[1, y, x] = y * 8 + 4
    return pixel_grid


def rotate_image(t, angle_, order, mode="constant", cval=0):
    t = t.permute(1, 2, 0).numpy()
    t = rotate(t, angle_, order=order, mode=mode, cval=cval)
    t = torch.from_numpy(t).permute(2, 0, 1).float()
    return t


def transform_kp(kp, max_size, image_ori, image_resize, angle, scale_factor):
    height = image_ori.shape[1]
    width = image_ori.shape[0]
    scale = max_size / min([height, width])
    # kp = kp * scale
    kp = kp * scale_factor
    # print(scale, scale_factor)

    h = image_resize.size(1)
    w = image_resize.size(2)

    translate = {"x": 0, "y": 0}

    shear = {"x": -0.0, "y": -0.0}
    scale = {"x": 1.0, "y": 1.0}

    rotate = -angle
    shift_x = w / 2 - 0.5
    shift_y = h / 2 - 0.5

    matrix_to_topleft = skimage.transform.SimilarityTransform(
        translation=[-shift_x, -shift_y]
    )
    matrix_shear_y_rot = skimage.transform.AffineTransform(rotation=-np.pi / 2)
    matrix_shear_y = skimage.transform.AffineTransform(shear=np.deg2rad(shear["y"]))
    matrix_shear_y_rot_inv = skimage.transform.AffineTransform(rotation=np.pi / 2)
    matrix_transforms = skimage.transform.AffineTransform(
        scale=(scale["x"], scale["y"]),
        translation=(translate["x"], translate["y"]),
        rotation=np.deg2rad(rotate),
        shear=np.deg2rad(shear["x"]),
    )
    matrix_to_center = skimage.transform.SimilarityTransform(
        translation=[shift_x, shift_y]
    )
    matrix = (
        matrix_to_topleft
        + matrix_shear_y_rot
        + matrix_shear_y
        + matrix_shear_y_rot_inv
        + matrix_transforms
        + matrix_to_center
    )

    kp2 = np.copy(kp)
    # kp2[:, [1, 0]] = kp2[:, [0, 1]]
    kp2 = np.expand_dims(kp2, 0)
    kp2 = cv2.transform(kp2, matrix.params[:2]).squeeze()
    # kp2[:, [1, 0]] = kp2[:, [0, 1]]

    # for idx in range(kp.shape[0]):
    #     y, x = kp[idx]
    #     x, y = cv2.transform(np.array([[[x, y]]]), matrix.params[:2]).squeeze()
    #     kp[idx] = [y, x]

    return kp2.astype(np.int32)


def transform_kp_aug_fast(
    kp_indices, image_height, scale_factor, image, image_transformed, angle
):
    keypoints = transform_kp(
        kp_indices,
        int(image_height * scale_factor),
        image,
        image_transformed,
        angle,
        scale_factor,
    )

    keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
    kp_map = np.zeros(
        [image_transformed.shape[1], image_transformed.shape[2]], dtype=np.int8
    )
    mask1 = np.bitwise_and(
        0 <= keypoints[:, 0], keypoints[:, 0] < image_transformed.shape[1]
    )
    mask2 = np.bitwise_and(
        0 <= keypoints[:, 1], keypoints[:, 1] < image_transformed.shape[2]
    )
    mask = np.bitwise_and(mask1, mask2)
    valid_keypoints = keypoints[mask]
    kp_map[valid_keypoints[:, 0], valid_keypoints[:, 1]] = 1
    return keypoints, valid_keypoints, kp_map, mask


def compute_attribution():
    from tqdm import trange
    from captum.attr import NeuronGradient

    m = DsacNet()
    m.cuda()
    inp = torch.randn((1, 1, 480, 640), requires_grad=True)
    inp = inp.cuda()

    neuron_deconv = NeuronGradient(m, m.fc3)
    data = {}
    for i in trange(60):
        for j in range(80):
            attribution = neuron_deconv.attribute(inp, (0, i, j))
            indices = torch.nonzero(attribution.squeeze() != 0).cpu().numpy()
            data[(i, j)] = indices
    ddir = "data/redkitchen/attribution.pkl"
    with open(ddir, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_attribution_slow(attribution, tree):
    bad_scene_points = []
    scores = []
    for scene_point in attribution:
        contribution = attribution[scene_point]
        dis, _ = tree.query(contribution)
        score = np.sum(dis == 0) / dis.shape[0]
        scores.append(score)
        if score == 0.0:
            bad_scene_points.append(scene_point)
    return bad_scene_points, scores


def process_attribution_fast(attribution, image_grid, point2indices, tree):
    bad_scene_points = []
    dis_arr, _ = tree.query(image_grid)
    dis_arr = dis_arr.astype(np.int32)
    scores = []
    for ind, scene_point in enumerate(attribution):
        indices = point2indices[scene_point]
        all_dis = dis_arr[indices]
        score = np.sum(all_dis == 0) / all_dis.shape[0]
        scores.append(score)
        # if (all_dis > 0).all():
        if score == 1:
            bad_scene_points.append(scene_point)
    return bad_scene_points


def return_heat_map(scene_heatmap2):
    scene_heatmap2 = normalize(1, 0, scene_heatmap2) * 255
    scene_heatmap2 = cv2.applyColorMap(
        scene_heatmap2.astype(np.uint8), cv2.COLORMAP_JET
    )
    return scene_heatmap2


def process_reproj_map(reproj_map):
    mask = reproj_map < 5
    non_zero = np.nonzero(mask)
    arr = np.vstack(non_zero).T
    tree = scipy_KDTree(arr)
    tree2 = scipy_KDTree(arr)
    reproj_map2 = np.ones_like(reproj_map)
    res = tree.query_ball_tree(tree2, 4)
    for idx in range(len(res)):
        if len(res[idx]) < 5:
            x, y = arr[idx]
            reproj_map2[x, y] = 0
    return reproj_map2


def compute_att_slow(i_arr, j_arr, attributor, attribution_default, image):
    ij2att = {}
    for i_ in range(i_arr.shape[0]):
        i = i_arr[i_]
        j = j_arr[i_]

        attribution = compute_att_slow_helper(
            attributor, image, [(0, i, j), (1, i, j), (2, i, j)]
        )
        indices = attribution_default[(i, j)]
        # attribution = normalize(
        #     0, 1, attribution[indices[:, 0], indices[:, 1]], using_pt=True
        # )
        # if torch.sum(torch.isnan(attribution_norm)).item() > 0:
        #     print()
        ij2att[(i, j)] = attribution
    return ij2att


def compute_att_slow_helper(attributor, image_, targets):
    res = None
    for target in targets:
        attribution = attributor.attribute(image_, target)
        if res is None:
            res = torch.abs(attribution)
        else:
            res += torch.abs(attribution)
        # if torch.sum(res) > 1e-5:
        #     break
    return res.squeeze()


def compute_att_fast_helper(attributor, image_, targets):
    res = None
    for target in targets:
        attribution = attributor.attribute(image_, target)
        if res is None:
            res = torch.abs(attribution)
        else:
            res += torch.abs(attribution)
    return res.squeeze()


def compute_att_fast(i_arr, j_arr, attributor, attribution_default, image, bs=32):
    image16 = torch.vstack([image for _ in range(bs)])
    ij2att = {}
    for i_ in range(0, i_arr.shape[0], bs):
        end_ = i_ + bs
        if end_ > i_arr.shape[0]:
            end_ = i_arr.shape[0]
            image_odd = torch.vstack([image for _ in range(end_ - i_)])
            targets = [
                [(0, i_arr[ind], j_arr[ind]) for ind in range(i_, end_)],
                [(1, i_arr[ind], j_arr[ind]) for ind in range(i_, end_)],
                [(2, i_arr[ind], j_arr[ind]) for ind in range(i_, end_)],
            ]
            attribution = compute_att_fast_helper(attributor, image_odd, targets)
        else:
            targets = [
                [(0, i_arr[ind], j_arr[ind]) for ind in range(i_, end_)],
                [(1, i_arr[ind], j_arr[ind]) for ind in range(i_, end_)],
                [(2, i_arr[ind], j_arr[ind]) for ind in range(i_, end_)],
            ]
            attribution = compute_att_fast_helper(attributor, image16, targets)

        target = targets[0]
        for ind_, (_, i, j) in enumerate(target):
            indices = attribution_default[(i, j)]
            att = attribution[ind_, :]
            att = normalize(0, 1, att[indices[:, 0], indices[:, 1]], using_pt=True)
            ij2att[(i, j)] = (indices, att)
    return ij2att


def check_square(indices):
    row2col = {}
    for i, j in indices:
        if i in row2col:
            row2col[i].append(j)
        else:
            row2col[i] = [j]
    print()


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def return_cam_mat():
    focal_length = 525.505
    cam_mat = torch.eye(3)
    cam_mat[0, 0] = focal_length
    cam_mat[1, 1] = focal_length
    cam_mat[0, 2] = 320
    cam_mat[1, 2] = 240
    return cam_mat


def sample_with_ic2avoid(ic2avoid, non_zero_ori, scene_coords_map):
    non_zero_ori_np = non_zero_ori.numpy()
    selected = set([])
    xy_selected = []
    for x, y in non_zero_ori_np:
        if (x, y) in ic2avoid:
            for k_ in ic2avoid[(x, y)]:
                old = len(selected)
                selected.add(k_)
                if len(selected) > old:
                    xy_selected.append((x, y))
    selected = np.array(list(selected))
    image_coordinates_gt = torch.from_numpy(np.array(xy_selected)).float().cuda()
    scene_coordinates_pred = scene_coords_map[:, :, selected[:, 0], selected[:, 1]][0]
    return image_coordinates_gt, scene_coordinates_pred


def max_ic2avoid(ic2avoid):
    res = {}
    for ic in ic2avoid:
        keys = list(ic2avoid[ic].keys())
        best = max(keys, key=lambda du: ic2avoid[ic][du])
        res[ic] = {best: ic2avoid[ic][best]}
    return res


def plot_histogram(scores):
    from matplotlib.pylab import plt

    q25, q75 = np.percentile(scores, [25, 75])
    bin_width = 2 * (q75 - q25) * len(scores) ** (-1 / 3)
    bins = round((scores.max() - scores.min()) / bin_width)
    plt.hist(scores, bins=bins)
    plt.show()
    # plt.close()


def reverse_attribution():
    afile = open("data/redkitchen/attribution.pkl", "rb")
    attribution_default = pickle.load(afile)
    afile.close()
    data = {}
    for sc_coord in attribution_default:
        for im_coord in attribution_default[sc_coord]:
            k1 = tuple(im_coord)
            data.setdefault(k1, set([])).add(sc_coord)
    all_im_coords = []
    for im_coord in data:
        all_im_coords.append(im_coord)
    all_im_coords = np.array(all_im_coords)
    tree = KDTree(all_im_coords)
    _, indices = tree.query(all_im_coords, 21)
    for i in tqdm(range(all_im_coords.shape[0])):
        im_coord = tuple(all_im_coords[i])
        neighbors = all_im_coords[indices[i, 1:]]
        for im_coord2 in neighbors:
            k2 = tuple(im_coord2)
            data[im_coord].update(data[k2])
    with open("data/redkitchen/imc2scc.pkl", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


def retrieve_gt_2d_3d_pairs(images_data, points_data, image_id):
    image = images_data[image_id]
    mask = image.point3D_ids >= 0
    im_coords = image.xys[mask]
    points = image.point3D_ids[mask]
    sc_coords = np.array([points_data[pid].xyz for pid in points])

    return im_coords, sc_coords
