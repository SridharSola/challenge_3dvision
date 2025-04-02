# further batch vectorised solution

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import roma
from PIL import Image
import pandas as pd
from challenge import load_rgb_depth, get_pos_rot, create_camera_gizmo
import torch

def create_point_cloud(depth_map, color_img, fov_deg, T=None):
    """Create a point cloud from depth map and color image.
    
    Args:
        depth_map: NxHxW depth map with metric distances
        color_img: NxHxWx3 RGB image
        fov_deg: horizontal field of view in degrees (N, )
        T: Nx4x4 transform matrix to apply to points (optional)
    
    Returns:
        points: NxDx3 array of transformed 3D points
        colors: NxDx3 array of colors normalized to [0,1]
        where D is H*W, num of pixels in the image
    """
    # Convert inputs to torch tensors if they aren't already
    if not isinstance(depth_map, torch.Tensor):
        depth_map = torch.from_numpy(depth_map).float()  # Convert to float32
    if not isinstance(color_img, torch.Tensor):
        color_img = torch.from_numpy(color_img).float()  
    if not isinstance(fov_deg, torch.Tensor):
        fov_deg = torch.from_numpy(fov_deg).float()    
    
    n, h, w = depth_map.shape
    
    # Create pixel coordinate grid
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), 
                           torch.arange(w, dtype=torch.float32), 
                           indexing='ij')
    
    # Convert to normalized image coordinates
    cx, cy = w/2, h/2
    # Convert fov_deg to (N,1,1) for broadcasting
    fx = (cx / torch.tan(torch.deg2rad(fov_deg/2)))[:, None, None]
    fy = fx
    
    # Expand xs and ys for broadcasting
    xs = xs[None, ...]  # (1,H,W)
    ys = ys[None, ...]  # (1,H,W)
    
    xs = (xs - cx) / fx
    ys = -(ys - cy) / fy
    
    # Create 3D points
    pts = torch.stack([
        xs * depth_map,
        ys * depth_map,
        -depth_map,
        torch.ones_like(depth_map)
    ], dim=1)  # (N,4,H,W)
    #print(pts.shape)
    
    # Reshape to (N,4,D)
    pts = pts.reshape(n, 4, -1)
    
    if T is not None:
        if not isinstance(T, torch.Tensor):
            T = torch.from_numpy(T).float()  
        pts = torch.matmul(T, pts)
    #print(pts.shape)
    
    # Convert from homogeneous coordinates
    pts = pts[:, :3] / pts[:, 3:] # (N,3,D)
    
    # Reshape to (N,D,3) and convert colors
    pts = pts.transpose(1, 2) # (N,D,3)
    print(pts.shape)
    colors = color_img.reshape(n, -1, 3) / 255.0
    
    return pts.numpy(), colors.numpy()

def project_points_to_image(points_3d, fov_deg, img_shape):
    """Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: (N, D, 3) array of 3D points in camera coordinates (Y-up)
        fov_deg: (N,) array of horizontal field of view in degrees
        img_shape: (height, width) of the image
    
    Returns:
        points_2d: (N, D, 2) array of image coordinates (Y-down)
        valid_mask: (N, D) boolean array indicating valid projections
    """
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.from_numpy(points_3d).float()
    if not isinstance(fov_deg, torch.Tensor):
        fov_deg = torch.from_numpy(fov_deg).float()
    
    h, w = img_shape
    cx, cy = w/2, h/2
    fx = (cx / torch.tan(torch.deg2rad(fov_deg/2)))[:, None]  # (N,1)
    fy = fx  # assume square pixels
    
    # Check for points in front of camera (negative Z in assignment's coordinate system)
    valid_mask = points_3d[..., 2] < 0  # (N,D)
    
    # Initialize projections
    points_2d = torch.zeros((*points_3d.shape[:-1], 2), dtype=torch.float32)  # (N,D,2)
    
    # Compute normalized coordinates for all points
    # Safe division - will be masked by valid_mask later
    points_normalized = points_3d[..., :2] / (-points_3d[..., 2:])  # (N,D,2)
    
    # Project all points and mask invalid ones later
    points_2d[..., 0] = points_normalized[..., 0] * fx + cx  # X remains same direction
    points_2d[..., 1] = h - (points_normalized[..., 1] * fy + cy)  # Y flipped
    
    # Mask invalid projections
    points_2d = torch.where(valid_mask[..., None], points_2d, 
                           torch.zeros_like(points_2d))
    
    return points_2d.numpy(), valid_mask.numpy()

def check_visibility(points_3d, depth_maps, fov_deg, img_shape, depth_threshold=0.1):
    """Check if 3D points are visible in camera views."""
    # Convert inputs to torch if needed
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.from_numpy(points_3d).float()
    if not isinstance(depth_maps, torch.Tensor):
        depth_maps = torch.from_numpy(depth_maps).float()
    
    # Project points to image coordinates
    points_2d, valid = project_points_to_image(points_3d, fov_deg, img_shape)
    points_2d = torch.from_numpy(points_2d).float()
    valid = torch.from_numpy(valid)
    
    # Initialize visibility mask
    visible = torch.zeros(points_3d.shape[:-1], dtype=torch.bool)
    
    # Get integer pixel coordinates
    px = torch.round(points_2d[..., 0]).long()  # (N,D)
    py = torch.round(points_2d[..., 1]).long()  # (N,D)
    
    # Filter points within image bounds
    in_bounds = (
        (px >= 0) & (px < img_shape[1]) &
        (py >= 0) & (py < img_shape[0]) &
        valid
    )  # (N,D)
    
    # For each batch
    for b in range(points_3d.shape[0]):
        if torch.any(in_bounds[b]):
            # Get valid pixel coordinates for this batch
            px_valid = px[b, in_bounds[b]]  # (M,)
            py_valid = py[b, in_bounds[b]]  # (M,)
            
            # Get depth values from depth map
            depth_vals = depth_maps[b, py_valid, px_valid]  # (M,)
            
            # Compare with point depths
            point_depths = -points_3d[b, in_bounds[b], 2]  # (M,)
            depth_diff = torch.abs(depth_vals - point_depths)
            
            # Points are visible if their depth matches the depth map
            matches = depth_diff < depth_threshold
            
            # Update visibility mask
            visible[b, in_bounds[b]] = matches
    
    return visible.numpy()

def check_occlusions(points_3d, depth_map, fov_deg, img_shape, depth_threshold=0.1):
    """Check which points are occluded in the target view.
    
    Args:
        points_3d: (N, 3) array of points in target camera coordinates
        depth_map: (H, W) depth map of target view
        fov_deg: field of view in degrees
        img_shape: (height, width) of target image
        depth_threshold: tolerance for depth comparison
        
    Returns:
        (N,) boolean array, True for non-occluded points
    """
    # Project points to get pixel coordinates
    points_2d, valid = project_points_to_image(points_3d, fov_deg, img_shape)
    
    # Initialize all points as occluded
    not_occluded = np.zeros(len(points_3d), dtype=bool)
    
    if np.any(valid):
        # Get integer pixel coordinates
        px = np.round(points_2d[valid, 0]).astype(int)
        py = np.round(points_2d[valid, 1]).astype(int)
        
        # Filter points within image bounds
        in_bounds = (
            (px >= 0) & (px < img_shape[1]) &
            (py >= 0) & (py < img_shape[0])
        )
        
        if np.any(in_bounds):
            # Get depth values from depth map
            px_valid = px[in_bounds]
            py_valid = py[in_bounds]
            depth_map_values = depth_map[py_valid, px_valid]
            
            # Get depths of projected points (negative because Z points into scene)
            point_depths = -points_3d[valid][in_bounds, 2]
            
            # Points are not occluded if they are close to the depth map value
            # and not significantly behind it
            depth_diff = point_depths - depth_map_values
            not_occluded_points = (np.abs(depth_diff) < depth_threshold) | (depth_diff < 0)
            
            # Update visibility mask
            valid_idx = np.where(valid)[0][in_bounds]
            not_occluded[valid_idx[not_occluded_points]] = True
    
    return not_occluded

def find_pixel_correspondences_batch(idx_pairs, visualize_steps=False):
    """Batch version of find_pixel_correspondences.
    
    Args:
        idx_pairs: (B,2) array of image index pairs
        visualize_steps: Whether to show intermediate visualizations
    
    Returns:
        tuple: (imgs_0, ps_0, imgs_1, ps_1) where:
            - imgs_0, imgs_1: (B,H,W,3) RGB images
            - ps_0, ps_1: List of (N_i,2) arrays containing corresponding pixel coordinates
    """
    batch_size = len(idx_pairs)
    meta = pd.read_csv(Path("dataset") / "data.csv")
    
    # Load all images and parameters in batch
    imgs_0 = []
    depths_0 = []
    imgs_1 = []
    depths_1 = []
    pos_0 = []
    rot_0 = []
    pos_1 = []
    rot_1 = []
    fov_0 = []
    fov_1 = []
    
    for i in range(batch_size):
        idx_0, idx_1 = idx_pairs[i]
        img_0, depth_0 = load_rgb_depth(idx_0)
        img_1, depth_1 = load_rgb_depth(idx_1)
        p0, r0 = get_pos_rot(meta.iloc[idx_0])
        p1, r1 = get_pos_rot(meta.iloc[idx_1])
        
        imgs_0.append(img_0)
        depths_0.append(depth_0)
        imgs_1.append(img_1)
        depths_1.append(depth_1)
        pos_0.append(p0)
        rot_0.append(r0)
        pos_1.append(p1)
        rot_1.append(r1)
        fov_0.append(meta.iloc[idx_0].cam_fov)
        fov_1.append(meta.iloc[idx_1].cam_fov)
    
    # Convert to arrays
    imgs_0 = np.stack(imgs_0)
    depths_0 = np.stack(depths_0)
    imgs_1 = np.stack(imgs_1)
    depths_1 = np.stack(depths_1)
    pos_0 = np.stack(pos_0)
    rot_0 = np.stack(rot_0)
    pos_1 = np.stack(pos_1)
    rot_1 = np.stack(rot_1)
    fov_0 = np.array(fov_0)
    fov_1 = np.array(fov_1)
    
    # Convert inputs to torch tensors
    depths_0 = torch.from_numpy(depths_0).float()
    imgs_0 = torch.from_numpy(imgs_0).float()  # Keep as uint8 for now
    depths_1 = torch.from_numpy(depths_1).float()
    imgs_1 = torch.from_numpy(imgs_1).float()  # Keep as uint8 for now
    fov_0 = torch.from_numpy(fov_0).float()
    fov_1 = torch.from_numpy(fov_1).float()
    
    # Create transformation matrices
    T_0 = torch.tile(torch.eye(4, dtype=torch.float32), (batch_size,1,1))
    T_0[...,:3,3] = torch.from_numpy(pos_0).float()
    R_0 = roma.unitquat_to_rotmat(torch.from_numpy(rot_0).float())
    T_0[...,:3,:3] = R_0
    
    T_1 = torch.tile(torch.eye(4, dtype=torch.float32), (batch_size,1,1))
    T_1[...,:3,3] = torch.from_numpy(pos_1).float()
    R_1 = roma.unitquat_to_rotmat(torch.from_numpy(rot_1).float())
    T_1[...,:3,:3] = R_1
    
    # Create point clouds and transform points in batch
    points_cam_a, colors_a = create_point_cloud(depths_0, imgs_0, fov_0, T_0)
    points_cam_a = torch.from_numpy(points_cam_a)
    
    # Transform points from Camera A to Camera B in batch
    T_1_inv = torch.inverse(T_1)
    points_cam_a_h = torch.cat([
        points_cam_a, 
        torch.ones((batch_size, points_cam_a.shape[1], 1))
    ], dim=-1)
    
    # Transform points using batch matrix multiplication
    points_cam_b = torch.matmul(T_1_inv, points_cam_a_h.transpose(-2,-1))
    points_cam_b = points_cam_b.transpose(-2,-1)
    points_cam_b = points_cam_b[..., :3] / points_cam_b[..., 3:]
    
    # Convert back to numpy only at the end
    points_cam_b = points_cam_b.numpy()
    
    # Process each batch for visibility and occlusion checks
    ps_0_list = []
    ps_1_list = []
    
    for b in range(batch_size):
        # Project and check visibility/occlusions
        points_img_b = project_points_to_image(points_cam_b[b:b+1], fov_1[b:b+1], imgs_1[b].shape[:2])[0][0]
        visible = check_visibility(points_cam_b[b:b+1], depths_1[b:b+1], fov_1[b:b+1], imgs_1[b].shape[:2])[0]
        
        # Get valid correspondences
        valid_points = visible & (
            (points_img_b[:,0] >= 0) & (points_img_b[:,0] < imgs_1[b].shape[1]) &
            (points_img_b[:,1] >= 0) & (points_img_b[:,1] < imgs_1[b].shape[0])
        )
        
        h, w = depths_0[b].shape
        ys, xs = np.meshgrid(range(h), range(w), indexing='ij')
        pixels_a = np.stack([xs.flatten(), ys.flatten()], axis=1)
        
        ps_0_list.append(pixels_a[valid_points].astype(np.int32))
        ps_1_list.append(points_img_b[valid_points].astype(np.int32))
    
    # Convert back to uint8 for visualization
    return (imgs_0.numpy().astype(np.uint8), 
            ps_0_list, 
            imgs_1.numpy().astype(np.uint8), 
            ps_1_list)

def main():
    """Main function to test the implementation."""
    # Test with multiple image pairs at once
    idx_pairs = np.array([
        [2, 4],
        [6, 9],
        [3, 5],
        [0, 1],
        [7, 8],
    ])
    # add timeit
    import time
    start_time = time.time()
    imgs_0, ps_0_list, imgs_1, ps_1_list = find_pixel_correspondences_batch(idx_pairs)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    # Visualize results
    for i in range(len(idx_pairs)):
        from challenge import visualize
        visualize(f"results/vec_result_{i}.jpg", imgs_0[i], imgs_1[i], ps_0_list[i], ps_1_list[i])
    
if __name__ == "__main__":
    main()
