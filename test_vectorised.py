import numpy as np
from solution import create_point_cloud as create_point_cloud_orig
from solution_vectorised import create_point_cloud as create_point_cloud_vec
from solution_vectorised import project_points_to_image as project_points_to_image_vec
from challenge import load_rgb_depth, get_pos_rot
import pandas as pd
from pathlib import Path
import roma
import torch


def reprojection_error_vectorised(idx_list, quantized=False):
    """Test reprojection error for batched implementation."""
    meta = pd.read_csv(Path("dataset") / "data.csv")
    
    # Batch load data
    depths = []
    imgs = []
    fovs = []
    
    for idx in idx_list:
        img, depth = load_rgb_depth(idx)
        
        if quantized:
            # Simulate 16-bit quantization like the original depth PNG
            min_depth = np.min(depth)
            max_depth = np.max(depth)
            depth_16bit = np.round((depth - min_depth) * 65535 / (max_depth - min_depth)).astype(np.uint16)
            depth = min_depth + (depth_16bit.astype(np.float64) * (max_depth - min_depth) / 65535)
        
        depths.append(depth)
        imgs.append(img)
        fovs.append(meta.iloc[idx].cam_fov)
    
    depths = np.stack(depths)
    imgs = np.stack(imgs)
    fovs = np.array(fovs)
    
    h, w = depths[0].shape
    
    # create point clouds from images
    points_cam, _ = create_point_cloud_vec(depths, imgs, fovs)
    
    # reproject to pixels
    points_2d, valid = project_points_to_image_vec(points_cam, fovs, (h, w))
    
    # original pixel grid
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    pixels = np.stack([xs.flatten(), ys.flatten()], axis=1)
    
    # Compute errors for each batch
    label = "Quantized" if quantized else "Float64"
    print(f"\nReprojection errors for vectorised implementation ({label}):")
    
    for b, idx in enumerate(idx_list):
        original_pixels = pixels[valid[b]]
        reprojected_pixels = points_2d[b][valid[b]]
        
        errors = np.linalg.norm(original_pixels - reprojected_pixels, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"Image {idx}:")
        print(f"  Mean error: {mean_error:.2e} pixels")
        print(f"  Max error: {max_error:.2e} pixels")


def compare_implementations(idx_list):
    """Compare results between original and vectorised implementations."""
    print("\nComparing original vs vectorised implementations:")
    
    from test import reprojection_error
    print("\nOriginal implementation:")
    for idx in idx_list:
        reprojection_error(idx, quantized=False)
    
    # vectorised implementation
    print("\nVectorised implementation:")
    reprojection_error_vectorised(idx_list, quantized=False)


if __name__ == "__main__":
    idx_list = [0, 1, 2, 3, 4]
    
    # Test vectorised implementation
    reprojection_error_vectorised(idx_list, quantized=False)
    reprojection_error_vectorised(idx_list, quantized=True)
    
    # Compare implementations
    #compare_implementations(idx_list) 