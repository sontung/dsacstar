import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import dsac_test
from dataset import CamLocDataset
from network_dsac import DsacNet
from utils import return_pixel_grid_dsac


def return_opt():
    parser = argparse.ArgumentParser(
        description="Initialize a scene coordinate regression network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("network", help="output file name for the network", default="")
    parser.add_argument("dataset", help="dataset name", default="redkitchen")

    parser.add_argument(
        "--image_dir", "-idir", type=str, default="..", help="dir for the images"
    )

    parser.add_argument(
        "--sfm_dir",
        "-sdir",
        type=str,
        default="../7scenes_reference_models",
        help="dir for sfm pgt models",
    )

    parser.add_argument(
        "--learningrate", "-lr", type=float, default=0.0001, help="learning rate"
    )

    parser.add_argument(
        "--iterations",
        "-iter",
        type=int,
        default=1000000,
        help="number of training iterations, i.e. numer of model updates",
    )

    parser.add_argument(
        "--inittolerance",
        "-itol",
        type=float,
        default=0.1,
        help="switch to reprojection error optimization when predicted scene coordinate is within this tolerance threshold to the ground truth scene coordinate, in meters",
    )

    parser.add_argument(
        "--mindepth",
        "-mind",
        type=float,
        default=0.1,
        help="enforce predicted scene coordinates to be this far in front of the camera plane, in meters",
    )

    parser.add_argument(
        "--maxdepth",
        "-maxd",
        type=float,
        default=1000,
        help="enforce that scene coordinates are at most this far in front of the camera plane, in meters",
    )

    parser.add_argument(
        "--targetdepth",
        "-td",
        type=float,
        default=10,
        help="if ground truth scene coordinates are unknown, use a proxy scene coordinate on the pixel ray with this distance from the camera, in meters",
    )

    parser.add_argument(
        "--softclamp",
        "-sc",
        type=float,
        default=100,
        help="robust square root loss after this threshold, in pixels",
    )

    parser.add_argument(
        "--hardclamp",
        "-hc",
        type=float,
        default=1000,
        help="clamp loss with this threshold, in pixels",
    )

    opt = parser.parse_args()
    return opt


def main_with_gt_keypoints(seed):
    start_time = time.time()
    debug_mode = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    opt = return_opt()
    if debug_mode:
        print("WARNING: RUNNING IN DEBUG MODE")

    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    out_dir = Path(f"data/{opt.dataset}/masked_pixels")
    out_dir.mkdir(parents=True, exist_ok=True)

    trainset = CamLocDataset(
        images_full_dir=opt.image_dir,
        dataset_name=opt.dataset,
        training=True,
        nb_preloaded_images=0 if debug_mode else 2000,
        nb_images=100 if debug_mode else -1,
        augment=True,
        using_filter=False,
        disable_coord_map=True,
        whole_image=False,
        return_true_pose=False,
    )

    trainset_loader = DataLoader(trainset, shuffle=True, num_workers=6, batch_size=1)

    # create network
    network = DsacNet()
    network = network.cuda()
    network.train()

    epochs = int(opt.iterations / len(trainset)) if not debug_mode else 1

    net_weights = f"{opt.network}-{opt.dataset}-mask=False"
    if Path(net_weights).is_file():
        print(f"Found stage 1 training weights, skipping training")
        network.load_state_dict(torch.load(net_weights))
    else:
        train_loop(
            network,
            trainset_loader,
            epochs,
            opt,
        )

    val_ds = CamLocDataset(
        images_full_dir=opt.image_dir,
        dataset_name=opt.dataset,
        training=False,
        augment=False,
        nb_preloaded_images=0,
        debug=True,
        using_filter=False,
        nb_images=100 if debug_mode else -1,
        disable_coord_map=True,
        return_true_pose=True,
    )
    val_ds_loader = DataLoader(val_ds, batch_size=6, shuffle=False, num_workers=4)
    score1 = dsac_test.evaluation_loop_full_dsac_pixel_grid_full(
        network, val_ds_loader, "cuda"
    )

    print(f"Took {(time.time()-start_time)/3600} hours")
    return score1,


def train_loop(network, trainset_loader, epochs, opt, using_masks=False):
    network.train()
    optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)
    iteration = 0
    pixel_grid = return_pixel_grid_dsac()

    for epoch in range(epochs):
        for sample in trainset_loader:
            optimizer.zero_grad()

            image = sample["image"]
            gt_pose = sample["pose"].float()

            # create camera calibration matrix
            focal_length = float(sample["focal"].item())
            cam_mat = torch.eye(3)
            cam_mat[0, 0] = focal_length
            cam_mat[1, 1] = focal_length
            cam_mat[0, 2] = image.size(3) / 2
            cam_mat[1, 2] = image.size(2) / 2
            cam_mat = cam_mat.cuda()

            scene_coords_map = network(image.cuda())
            assert scene_coords_map.shape[0] == 1

            scene_coordinates_pred = scene_coords_map.squeeze().view(3, -1)
            scene_coords = torch.vstack(
                [
                    scene_coordinates_pred,
                    torch.ones((1, scene_coordinates_pred.size(1)), device="cuda"),
                ]
            )

            pixel_grid_crop = pixel_grid[
                :, 0 : scene_coords_map.size(2), 0 : scene_coords_map.size(3)
            ].clone()
            pixel_grid_crop = pixel_grid_crop.view(2, -1).cuda()

            # prepare pose for projection operation
            gt_pose = gt_pose[0][0:3, :]
            gt_pose = gt_pose.cuda()

            # scene coordinates to camera coordinate
            camera_coords = torch.mm(gt_pose, scene_coords)

            # re-project predicted scene coordinates
            reprojection_error = torch.mm(cam_mat, camera_coords)
            reprojection_error[2].clamp_(min=opt.mindepth)  # avoid division by zero
            reprojection_error = reprojection_error[0:2] / reprojection_error[2]
            reprojection_error = reprojection_error - pixel_grid_crop
            reprojection_error = reprojection_error.norm(2, 0)

            # check predicted scene coordinate for various constraints
            invalid_min_depth = (
                camera_coords[2] < opt.mindepth
            )  # behind or too close to camera plane
            invalid_repro = (
                reprojection_error > opt.hardclamp
            )  # check for very large reprojection errors
            # no ground truth scene coordinates available, enforce max distance of predicted coordinates
            invalid_max_depth = camera_coords[2] > opt.maxdepth

            # combine all constraints
            valid_scene_coordinates = (
                invalid_min_depth + invalid_max_depth + invalid_repro
            ) == 0
            num_valid_sc = int(valid_scene_coordinates.sum())
            if num_valid_sc > 0:

                reprojection_error = reprojection_error[valid_scene_coordinates]

                # calculate soft clamped l1 loss of reprojection error
                loss_l1 = reprojection_error[reprojection_error <= opt.softclamp]
                loss_sqrt = reprojection_error[reprojection_error > opt.softclamp]
                loss_sqrt = torch.sqrt(opt.softclamp * loss_sqrt)

                loss = loss_l1.sum() + loss_sqrt.sum()

                if num_valid_sc < scene_coords.size(1):
                    invalid_scene_coordinates = valid_scene_coordinates == 0

                    # generate proxy coordinate targets with constant depth assumption
                    target_camera_coords = pixel_grid_crop
                    target_camera_coords[0] -= image.size(3) / 2
                    target_camera_coords[1] -= image.size(2) / 2
                    target_camera_coords *= opt.targetdepth
                    target_camera_coords /= focal_length
                    # make homogeneous
                    target_camera_coords = torch.cat(
                        (
                            target_camera_coords,
                            torch.ones((1, target_camera_coords.size(1))).cuda(),
                        ),
                        0,
                    )

                    # distance
                    loss += torch.abs(
                        camera_coords[:, invalid_scene_coordinates]
                        - target_camera_coords[:, invalid_scene_coordinates]
                    ).sum()

                loss /= scene_coords.size(1)
                num_valid_sc /= scene_coords.size(1)
                loss.backward()  # calculate gradients (pytorch autograd)
                optimizer.step()  # update all model parameters
                optimizer.zero_grad()

            iteration = iteration + 1

        torch.save(
            network.state_dict(), f"{opt.network}-{opt.dataset}-mask={using_masks}"
        )


if __name__ == "__main__":
    # example params
    # checkpoints/net pumpkin --image_dir ../7_scenes
    seeds = [14, 4, 95]
    main_with_gt_keypoints(seeds[0])
