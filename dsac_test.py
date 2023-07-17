import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CamLocDataset
from utils import (
    evaluate_pose_quality,
    return_pixel_grid_dsac,
)


def evaluation_loop_full_dsac_pixel_grid_full(
    model, val_ds_loader, device, file_name="results/results_pred_d2.txt"
):
    model.eval()
    ind = 0

    stats = {
        "pose_error_pred": [],
        "inlier_pred": [],
        "rot_error": [],
        "trans_error": [],
    }

    dsac_pixels_grid = return_pixel_grid_dsac()
    pixel_grid_crop = None
    with torch.no_grad():
        for sample in val_ds_loader:
            ind += 1
            image = sample["image"].to(device)
            scene_map_pred = model(image)
            if pixel_grid_crop is None:
                pixel_grid_crop = dsac_pixels_grid[
                    :, 0 : scene_map_pred.size(2), 0 : scene_map_pred.size(3)
                ].clone()
                pixel_grid_crop = pixel_grid_crop.view(2, -1)
                pixel_grid_crop = pixel_grid_crop.permute([1, 0])
            bs = scene_map_pred.size(0)
            scene_map_pred = scene_map_pred.view(bs, 3, -1)

            for batch_index in range(scene_map_pred.shape[0]):
                scene_coordinate = scene_map_pred[batch_index]
                scene_coordinate = scene_coordinate.permute([1, 0])
                error, r_err, t_err = evaluate_pose_quality(
                    pixel_grid_crop.cpu().numpy(),
                    scene_coordinate.cpu().numpy(),
                    sample["params"][batch_index],
                    sample["pose"][batch_index],
                    return_errors_only=True,
                )

                stats["pose_error_pred"].append(error)
                stats["rot_error"].append(r_err)
                stats["trans_error"].append(t_err)
                # qw, qx, qy, qz = pose_pred.q
                # tx, ty, tz = pose_pred.t
                # result1 = f"{sample['image_name'][batch_index]} {qw} {qx} {qy} {qz} {tx} {ty} {tz}"
                # print(result1, file=test_file_result)

    verbose = {}
    for k in list(stats.keys()):
        if len(stats[k]) > 0:
            verbose[k] = np.median(stats[k])
            print(k, verbose[k])
    pose_errors = np.array(stats["pose_error_pred"])
    print(pose_errors[pose_errors < 5].shape[0] / pose_errors.shape[0])
    return pose_errors[pose_errors < 5].shape[0] / pose_errors.shape[0]


def main(
    weights_dir="models/init_net_gt",
    network=None,
    file_name="results/results_pred_d2.txt",
):
    device = "cuda"
    if network is None:
        from network_dsac import DsacNet

        network = DsacNet()
        state_dict = torch.load(weights_dir)
        network.load_state_dict(state_dict)
        network.to(device)

    val_ds = CamLocDataset(
        images_full_dir="../7_scenes",
        dataset_name="redkitchen",
        training=False,
        augment=False,
        nb_preloaded_images=0,
        debug=True,
        using_filter=False,
        nb_images=-1,
        disable_coord_map=True,
        return_true_pose=True,
    )
    val_ds_loader = DataLoader(val_ds, batch_size=6, shuffle=False, num_workers=4)
    evaluation_loop_full_dsac_pixel_grid_full(network, val_ds_loader, device)


if __name__ == "__main__":
    main(
        "checkpoints/net-redkitchen-mask=True",
        file_name="results/results_pred_mse.txt",
    )
