import argparse
import random
import time
from pathlib import Path
import dsacstar
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

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
            gt_pose = gt_pose[0][0:3, :]

            # create camera calibration matrix
            focal_length = float(sample["focal"].item())
            cam_mat = torch.eye(3)
            cam_mat[0, 0] = focal_length
            cam_mat[1, 1] = focal_length
            cam_mat[0, 2] = image.size(3) / 2
            cam_mat[1, 2] = image.size(2) / 2
            cam_mat = cam_mat.cuda()

            scene_coordinates = network(image.cuda())
            scene_coordinate_gradients = torch.zeros(scene_coordinates.size())

            loss = dsacstar.backward_rgb(
                scene_coordinates.cpu(),
                scene_coordinate_gradients,
                gt_pose,
                opt.hypotheses,
                opt.threshold,
                focal_length,
                float(image.size(3) / 2),  # principal point assumed in image center
                float(image.size(2) / 2),
                opt.weightrot,
                opt.weighttrans,
                opt.softclamp,
                opt.inlieralpha,
                opt.maxpixelerror,
                network.OUTPUT_SUBSAMPLE,
                random.randint(0, 1000000))  # used to initialize random number generator in C++

            # update network parameters
            torch.autograd.backward((scene_coordinates), (scene_coordinate_gradients.cuda()))
            optimizer.step()
            optimizer.zero_grad()

            print('Iteration: %6d, Loss: %.2f \n' % (iteration, loss), flush=True)

            iteration = iteration + 1

    torch.save(
        network.state_dict(), f"{opt.network}-{opt.dataset}-e2e-mask={using_masks}"
    )


if __name__ == "__main__":
    # example params
    # checkpoints/net pumpkin --image_dir ../7_scenes
    seeds = [14, 4, 95]
    main_with_gt_keypoints(seeds[0])
