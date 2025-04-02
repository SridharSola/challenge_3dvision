from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import roma
from PIL import Image
import pandas as pd
from challenge import load_rgb_depth, get_pos_rot, create_camera_gizmo
import torch

def create_point_cloud(depth_map, color_img, fov_deg, T=np.eye(4)):
    """Create a point cloud from depth map and color image.
    
    Args:
        depth_map: HxW depth map with metric distances
        color_img: HxWx3 RGB image
        fov_deg: horizontal field of view in degrees
        T: 4x4 transform matrix to apply to points
    
    Returns:
        open3d.geometry.PointCloud
    """
    h, w = depth_map.shape
    
    # Create pixel coordinate grid
    ys, xs = np.meshgrid(range(h), range(w), indexing='ij')
    
    # Convert to normalized image coordinates
    cx, cy = w/2, h/2
    fx = cx / np.tan(np.deg2rad(fov_deg/2))  # focal length from FoV
    fy = fx  # assume square pixels
    
    xs = (xs - cx) / fx
    ys = -(ys - cy) / fy  # Flip Y to match camera coordinate system (Y-up)
    
    # Create 3D points
    pts = np.stack([
        xs * depth_map,  # X right
        ys * depth_map,  # Y up
        -depth_map,      # Z into scene (negative)
        np.ones_like(depth_map)
    ])
    
    # Reshape to 4xN
    pts = pts.reshape(4, -1)
    
    # Transform points
    pts = T @ pts
    pts = pts[:3] / pts[3:]  # Convert from homogeneous
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.T)
    
    # Add colors
    colors = color_img.reshape(-1, 3) / 255.0  # normalize to [0,1]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def project_points_to_image(points_3d, fov_deg, img_shape):
    """Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: (N, 3) array of 3D points in camera coordinates (Y-up)
        fov_deg: horizontal field of view in degrees
        img_shape: (height, width) of the image
    
    Returns:
        points_2d: (N, 2) array of image coordinates (Y-down)
        valid_mask: (N,) boolean array indicating valid projections
    """
    h, w = img_shape
    cx, cy = w/2, h/2
    fx = cx / np.tan(np.deg2rad(fov_deg/2))
    fy = fx
    
    # Check for points in front of camera (negative Z in assignment's coordinate system)
    valid_mask = points_3d[:, 2] < 0
    
    # Initialize projections
    points_2d = np.zeros((len(points_3d), 2))
    
    if np.any(valid_mask):
        # First do the perspective division
        points_normalized = points_3d[valid_mask, :2] / (-points_3d[valid_mask, 2:])
        
        # Project valid points
        # X remains the same direction (right)
        points_2d[valid_mask, 0] = points_normalized[:, 0] * fx + cx
        
        # Y needs to be flipped for image coordinates (Y-down)
        points_2d[valid_mask, 1] = h - (points_normalized[:, 1] * fy + cy)
    
    return points_2d, valid_mask

def check_visibility(points_3d, depth_map, fov_deg, img_shape, depth_threshold=0.1):
    """Check if 3D points are visible in a camera view.
    
    Args:
        points_3d: (N, 3) array of points in camera coordinates
        depth_map: (H, W) depth image
        fov_deg: camera field of view in degrees
        img_shape: (height, width) of image
        depth_threshold: max allowed difference between point and depth map
    
    Returns:
        (N,) boolean array indicating visible points
    """
    # Project points to image coordinates
    points_2d, valid = project_points_to_image(points_3d, fov_deg, img_shape)
    
    # Initialize visibility mask
    visible = np.zeros(len(points_3d), dtype=bool)
    
    # Check only valid projections
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
            depth_vals = depth_map[py_valid, px_valid]
            
            # Compare with point depths
            point_depths = -points_3d[valid][in_bounds, 2]  # negative because z points into scene
            depth_diff = np.abs(depth_vals - point_depths)
            
            # Points are visible if their depth matches the depth map
            matches = depth_diff < depth_threshold
            
            # Update visibility mask
            valid_idx = np.where(valid)[0][in_bounds]
            visible[valid_idx[matches]] = True
    
    return visible

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

def find_pixel_correspondences(idx_0: int, idx_1: int, visualize_steps: bool = False):
    """
    Find corresponding pixels between two images.
    
    Args:
        idx_0: Index of the first image
        idx_1: Index of the second image
        visualize_steps: Whether to show intermediate visualizations
        
    Returns:
        tuple: (img_0, ps_0, img_1, ps_1) where:
            - img_0, img_1: RGB images
            - ps_0, ps_1: Arrays of shape (N, 2) containing corresponding pixel coordinates
    """
    # Load metadata
    meta = pd.read_csv(Path("dataset") / "data.csv")
    meta_0 = meta.iloc[idx_0]
    meta_1 = meta.iloc[idx_1]
    
    # Load images and depth maps using helper function
    img_0, depth_0 = load_rgb_depth(idx_0)
    img_1, depth_1 = load_rgb_depth(idx_1)
    
    # Get camera parameters using helper function
    pos_0, rot_0 = get_pos_rot(meta_0)
    pos_1, rot_1 = get_pos_rot(meta_1)
    
    # Store image dimensions
    h, w = img_0.shape[:2]
    
    # Visualize the loaded data
    if visualize_steps:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(img_0)
        axs[0, 0].set_title("Image A (RGB)")
        axs[0, 1].imshow(depth_0)
        axs[0, 1].set_title("Image A (Depth)")
        axs[1, 0].imshow(img_1)
        axs[1, 0].set_title("Image B (RGB)")
        axs[1, 1].imshow(depth_1)
        axs[1, 1].set_title("Image B (Depth)")
        plt.tight_layout()
        plt.show()

    # Create pixel coordinate grid for Image A
    h, w = depth_0.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    # Don't flip Y-axis for visualization since we already flipped in create_point_cloud
    pixels_a_vis = np.stack([xs.flatten(), ys.flatten()], axis=1)
    pixels_a = np.stack([xs.flatten(), ys.flatten()], axis=1)

    # Create point cloud from Image A
    pcd_0 = create_point_cloud(depth_0, img_0, meta_0.cam_fov)
    points_cam_a = np.asarray(pcd_0.points)

    # Create transformation matrices ensuring consistent coordinate systems
    T_0 = np.eye(4)
    T_0[:3, 3] = pos_0
    R_0 = roma.unitquat_to_rotmat(torch.from_numpy(rot_0)).numpy()
    # Ensure right-handed system with Y-up and Z-negative
    T_0[:3, :3] = R_0
    
    T_1 = np.eye(4)
    T_1[:3, 3] = pos_1
    R_1 = roma.unitquat_to_rotmat(torch.from_numpy(rot_1)).numpy()
    # Ensure right-handed system with Y-up and Z-negative
    T_1[:3, :3] = R_1
    

    # Transform points from Camera A to Camera B
    # First transform from Camera A to world coordinates
    points_world = (T_0 @ np.hstack([points_cam_a, np.ones((points_cam_a.shape[0], 1))]).T).T[:, :3]
    
    # Then transform from world to Camera B coordinates using inverse transform
    T_1_inv = np.linalg.inv(T_1)  # We need inverse to go from world to camera B
    points_cam_b = (T_1_inv @ np.hstack([points_world, np.ones((points_world.shape[0], 1))]).T).T[:, :3]

    # Project points onto Image B plane
    points_img_b, valid_b = project_points_to_image(points_cam_b, meta_1.cam_fov, img_1.shape[:2])

    # Check visibility and occlusions in Image B
    visible_in_b = check_visibility(points_cam_b, depth_1, meta_1.cam_fov, img_1.shape[:2], depth_threshold=0.1)
    not_occluded = check_occlusions(points_cam_b, depth_1, meta_1.cam_fov, img_1.shape[:2], depth_threshold=0.1)

    # Combined validity mask including occlusion check
    valid_points = (
        valid_b &
        visible_in_b &
        not_occluded &  # Add occlusion check
        (points_img_b[:, 0] >= 0) & (points_img_b[:, 0] < img_1.shape[1]) &
        (points_img_b[:, 1] >= 0) & (points_img_b[:, 1] < img_1.shape[0])
    )

    # Get corresponding points using visualization-corrected coordinates for Image A
    ps_0 = pixels_a_vis[valid_points].astype(np.int32)  # Corrected visualization coordinates
    ps_1 = points_img_b[valid_points].astype(np.int32)

    # Calculate polar coordinates based on corrected Image A coordinates
    center_y, center_x = h/2, w/2
    angles = np.arctan2(ps_0[:, 1] - center_y, ps_0[:, 0] - center_x)
    hues = (angles + np.pi) / (2 * np.pi)

    # DEBUG: Visualize camera coordinate systems and point clouds
    #print("\nCamera Coordinate Systems:")
    #print(f"Camera A position: {pos_0}")
    #print(f"Camera A rotation (quaternion): {rot_0}")
    #print(f"Camera B position: {pos_1}")
    #print(f"Camera B rotation (quaternion): {rot_1}")
    
    # Create coordinate frame visualizations with labels
    size = 0.5
    coord_frame_a = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    coord_frame_b = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    # Create camera frustums with smaller size
    frustum_a = create_camera_gizmo(T_0, meta_0.cam_fov, img_0.shape[:2], frustum_distance=0.5)
    frustum_b = create_camera_gizmo(T_1, meta_1.cam_fov, img_1.shape[:2], frustum_distance=0.5)
    
    # Create point clouds in world coordinates
    pcd_0.transform(T_0)  # Transform Camera A points to world
    pcd_1 = create_point_cloud(depth_1, img_1, meta_1.cam_fov)
    pcd_1.transform(T_1)  # Transform Camera B points to world
    
    # Paint point clouds different colors to distinguish them
    pcd_0.paint_uniform_color([1, 0.7, 0.7])  # Light red for Camera A
    pcd_1.paint_uniform_color([0.7, 0.7, 1])  # Light blue for Camera B
    
    # Create text labels for axes
    labels = []
    for i, (text, pos, color) in enumerate([
        ('X', [size, 0, 0], [1, 0, 0]),
        ('Y', [0, size, 0], [0, 1, 0]),
        ('Z', [0, 0, size], [0, 0, 1])
    ]):
        # Labels for Camera A
        label_a = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        label_a.paint_uniform_color(color)
        label_a.translate(pos)
        labels.append(label_a)
        
        # Labels for Camera B
        label_b = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        label_b.paint_uniform_color(color)
        label_b.translate(pos)
        labels.append(label_b)
    
    # Transform coordinate frames to camera positions
    coord_frame_a.transform(T_0)
    coord_frame_b.transform(T_1)
    
    # Transform labels
    for i in range(0, len(labels), 2):
        labels[i].transform(T_0)     # Camera A labels
        labels[i+1].transform(T_1)   # Camera B labels
    
    # Add text to visualize which is which
    #print("Red = X-axis (right)")
    #print("Green = Y-axis (up)")
    #print("Blue = Z-axis (into scene, negative)")
    #print("Light red points = Camera A point cloud")
    #print("Light blue points = Camera B point cloud")
    
    # Visualize everything in Open3D
    geometries = [
        coord_frame_a, coord_frame_b,  # Coordinate frames
        pcd_0, pcd_1,                  # Point clouds
        *frustum_a, *frustum_b,        # Camera frustums
        *labels                        # Axis labels
    ]
    if visualize_steps:
        o3d.visualization.draw_geometries(geometries)

    # Remove the duplicate matplotlib visualization and keep only the challenge.py one
    return img_0, ps_0, img_1, ps_1

def main():
    """Main function to test the implementation."""
    # Test with random image pair
    sample_idxs = [
        [2, 4],
        [6, 9],
        [3, 5],
        [0, 1],
        [7, 8],
    ]
    for idx_0, idx_1 in sample_idxs:
        img_0, ps_0, img_1, ps_1 = find_pixel_correspondences(idx_0, idx_1)
        # Use the visualization function from challenge.py
        from challenge import visualize
        visualize("test_result.jpg", img_0, img_1, ps_0, ps_1)
    

if __name__ == "__main__":
    main()
