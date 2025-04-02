import numpy as np
from solution import create_point_cloud, project_points_to_image
from challenge import load_rgb_depth, get_pos_rot
import pandas as pd
from pathlib import Path
import roma
import torch


def reprojection_error(idx, quantized=False):
    meta = pd.read_csv(Path("dataset") / "data.csv")
    meta_row = meta.iloc[idx]
    img, depth = load_rgb_depth(idx)
    pos, rot = get_pos_rot(meta_row)

    h, w = depth.shape

    if quantized:
        # Simulate 16-bit quantization like the original depth PNG
        min_depth = np.min(depth)
        max_depth = np.max(depth)
        depth_16bit = np.round((depth - min_depth) * 65535 / (max_depth - min_depth)).astype(np.uint16)
        depth = min_depth + (depth_16bit.astype(np.float64) * (max_depth - min_depth) / 65535)

    # Create point cloud from image
    pcd = create_point_cloud(depth, img, meta_row.cam_fov)
    points_cam = np.asarray(pcd.points)

    # Reproject to pixels
    points_2d, valid = project_points_to_image(points_cam, meta_row.cam_fov, (h, w))

    # Original pixel grid
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    original_pixels = np.stack([xs.flatten(), ys.flatten()], axis=1)[valid]
    reprojected_pixels = points_2d[valid]

    # Compute reprojection error
    errors = np.linalg.norm(original_pixels - reprojected_pixels, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    label = "Quantized" if quantized else "Float64"
    print(f"Reprojection error for image {idx} ({label}):")
    print(f"  Mean error: {mean_error:.2e} pixels")
    print(f"  Max error: {max_error:.2e} pixels\n")


if __name__ == "__main__":
    for idx in [0, 1, 2, 3, 4]:
        reprojection_error(idx, quantized=False)
        reprojection_error(idx, quantized=True)
