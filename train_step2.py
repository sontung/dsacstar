import argparse
import random
import time
from pathlib import Path
import torch
import sys
from tqdm import tqdm

import dsacstar
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CamLocDataset
from network_dsac import DsacNet
from skimage.transform import rotate as ski_rotate
from skimage.transform import resize as ski_resize


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
        "--mask", "-mask", type=bool, default=False, help="using masks"
    )

    parser.add_argument(
        "--maskdir", "-mdir", type=str, default="../vloc-optimization-understand/data", help="using masks"
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
        default=10000,
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

    parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                        help='number of hypotheses, i.e. number of RANSAC iterations')

    parser.add_argument('--threshold', '-t', type=float, default=10,
                        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                        help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

    parser.add_argument('--weightrot', '-wr', type=float, default=1.0,
                        help='weight of rotation part of pose loss')

    parser.add_argument('--weighttrans', '-wt', type=float, default=100.0,
                        help='weight of translation part of pose loss')

    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

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
        return_true_pose=True,
    )

    trainset_loader = DataLoader(trainset, shuffle=True, num_workers=6, batch_size=1)

    # create network
    network = DsacNet()
    network = network.cuda()
    network.train()

    epochs = int(opt.iterations / len(trainset)) if not debug_mode else 1

    net_weights = f"{opt.network}-{opt.dataset}-mask=False"
    if Path(net_weights).is_file():
        print(f"Found stage 1 training weights at {net_weights}")
        if opt.mask:
            print(f"Mask dir is at {opt.maskdir}")
        network.load_state_dict(torch.load(net_weights))
        train_loop(
            network,
            trainset_loader,
            epochs,
            opt,
            opt.mask,
        )
    else:
        if debug_mode:
            loop2(
                network,
                trainset,
                epochs,
                opt,
                opt.mask,
            )
        else:
            print(f"Init weights not found at {net_weights}")


def loop2(network, trainset_loader, epochs, opt, using_masks=False):
    network.train()
    epochs = 100
    pbar = tqdm(total=epochs*len(trainset_loader), desc="Training")

    for epoch in range(10):
        for sample in trainset_loader:
            pbar.update(1)


def train_loop(network, trainset_loader, epochs, opt, using_masks=False):
    network.train()
    optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)
    iteration = 0
    epochs = 100
    pbar = tqdm(total=epochs*len(trainset_loader), desc="Training")

    for epoch in range(epochs):
        for sample in trainset_loader:
            optimizer.zero_grad()

            image = sample["image"]
            gt_pose = sample["pose"].float()
            gt_pose = gt_pose[0]

            focal_length = float(sample["focal"].item())

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

            if using_masks:
                mask = np.load(
                    f"{opt.maskdir}/{opt.dataset}/masked_pixels/{sample['image_id'][0].item()}.npy"
                )
                mask = augment_mask(mask, sample["angle"][0], scene_coordinates.shape)
                mask = torch.tensor(mask, dtype=torch.bool)
                scene_coordinate_gradients[:, :, mask] = 0

            # update network parameters
            torch.autograd.backward((scene_coordinates), (scene_coordinate_gradients.cuda()))
            optimizer.step()
            optimizer.zero_grad()
            iteration = iteration + 1
            pbar.update(1)

        torch.save(
            network.state_dict(), f"checkpoints/net-{opt.dataset}-e2e-mask={using_masks}"
        )


def augment_mask(mask, angle, shape):
    mask = mask.reshape(60, 80).astype(np.uint8)
    mask = ski_resize(mask, shape[2:], order=0)
    if angle != 0:
        mask = ski_rotate(mask, angle, order=0, mode="constant", cval=0)
    # mask = mask.reshape(-1)
    return mask


if __name__ == "__main__":
    # example params
    # checkpoints/net pumpkin --image_dir ../7_scenes
    seeds = [14, 4, 95]
    main_with_gt_keypoints(seeds[0])
