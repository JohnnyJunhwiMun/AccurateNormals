"""
#===============================================================================
# PROGRAM OVERVIEW
#===============================================================================
#
# This program analyzes surface normal vectors using a RealSense depth camera.
# It processes recorded .bag files to detect markers, calculate and compare 
# normal vectors around these markers, and evaluate the consistency of the 
# angle between these vectors.
#
# The program has three main phases:
# 1. Initial stabilization - Allows the system to stabilize before measurements
# 2. Parameter optimization - Tests different smoothing parameters to find optimal settings
# 3. Sampling - Records and analyzes data with the optimal parameters
#
# Key features:
# - Detects red markers using HSV color filtering
# - Calculates surface normal vectors using PCA
# - Applies Gaussian smoothing and voxel grid filtering to reduce noise
# - Monitors angle differences between normal vectors
# - Visualizes results in real-time and generates summary statistics
#
# PROGRAM OUTLINE:
# 
# 1. UTILITY FUNCTIONS
#    - Point cloud processing (Gaussian filter, voxel grid filter)
#    - Normal vector calculation with PCA
#
# 2. INITIALIZATION
#    - UI setup (HSV trackbars for color filtering)
#    - RealSense pipeline configuration
#    - Parameter optimization setup
#    - Variables for testing phases 
#    - Visualization setup
#
# 3. MAIN PROCESSING LOOP
#    - Frame acquisition from .bag file
#    - Image processing and marker detection
#    - Normal vector calculation
#    - Parameter optimization phase
#    - Post-stabilization phase 
#    - Sampling phase
#
# 4. RESULTS PROCESSING
#    - Statistics calculation
#    - Summary plot generation
#
# Usage: Run the program with a connected RealSense camera or a pre-recorded
# .bag file. Adjust the HSV trackbars to isolate red markers in the image.
#
# Requirements: pyrealsense2, numpy, opencv-python, scipy, matplotlib
#
"""

import pyrealsense2 as rs  # Intel RealSense camera library for depth sensing
import numpy as np  # For numerical operations and array handling
import cv2  # OpenCV for image processing and computer vision
from scipy.spatial import KDTree  # For efficient nearest-neighbor search
from collections import deque  # For maintaining fixed-size buffers (history)
import matplotlib.pyplot as plt  # For visualization and plotting
import datetime  # For timestamp generation and time-based operations

#===============================================================================
# UTILITY FUNCTIONS
#===============================================================================

def nothing(x):
    """Dummy function for OpenCV trackbar callbacks"""
    pass

#-------------------------------------------------------------------------------
# Point Cloud Processing Functions
#-------------------------------------------------------------------------------

def gaussian_weights(distances, sigma):
    """
    Calculate Gaussian weights based on distances and sigma parameter.
    
    Args:
        distances: Array of distances between points
        sigma: Gaussian kernel width parameter
    
    Returns:
        Weights calculated using Gaussian function
    """
    return np.exp(- (distances ** 2) / (2 * sigma ** 2))

def gaussian_filter_point_cloud(points, k, sigma, distances_all=None, indices_all=None, weights_cache=None):
    """
    Apply Gaussian filtering to a point cloud.
    
    This smooths the point cloud by replacing each point with a weighted average
    of its neighbors, where weights are determined by a Gaussian function of distance.
    
    Args:
        points: Input point cloud (Nx3 array)
        k: Number of neighbors to consider for each point
        sigma: Gaussian kernel width parameter (controls smoothing strength)
        distances_all: Pre-computed distances (optional, for efficiency)
        indices_all: Pre-computed neighbor indices (optional, for efficiency)
        weights_cache: Cache of pre-computed weights (optional, for efficiency)
    
    Returns:
        Smoothed point cloud
    """
    if sigma is None or sigma == "None":
        return points
    if weights_cache is None:
        weights_cache = {}
    num_points = points.shape[0]
    k_safe = min(k, num_points)
    if distances_all is None or indices_all is None:
        tree = KDTree(points)
        distances_all, indices_all = tree.query(points, k=k_safe)
    if distances_all.ndim == 1:
        distances_all = distances_all[:, np.newaxis]
    if indices_all.ndim == 1:
        indices_all = indices_all[:, np.newaxis]
    if (k_safe, sigma) not in weights_cache:
        distances = distances_all[:, :k_safe]
        sigma_val = float(sigma)
        w = gaussian_weights(distances, sigma_val)
        sum_w = np.sum(w, axis=1)
        sum_w[sum_w == 0] = 1
        w /= sum_w[:, np.newaxis]
        weights_cache[(k_safe, sigma)] = w
    else:
        w = weights_cache[(k_safe, sigma)]
    actual_neighbors = indices_all.shape[1]
    if actual_neighbors < k_safe:
        k_safe = actual_neighbors
    neighbor_points = points[indices_all[:, :k_safe]]
    smoothed_points = np.sum(neighbor_points * w[:, :k_safe, np.newaxis], axis=1)
    return smoothed_points

def voxel_grid_filter(points, voxel_size):
    """
    Apply voxel grid filtering to reduce point cloud density.
    
    Divides 3D space into voxels and replaces all points within each voxel
    with their centroid, effectively downsampling the point cloud while
    preserving its overall structure.
    
    Args:
        points: Input point cloud (Nx3 array)
        voxel_size: Size of each voxel (in meters)
    
    Returns:
        Downsampled point cloud
    """
    if len(points) == 0:
        return points
    discretized = np.floor(points / voxel_size)
    voxel_dict = {}
    for i, coord in enumerate(discretized):
        key = tuple(coord.astype(np.int32))
        if key in voxel_dict:
            voxel_dict[key].append(points[i])
        else:
            voxel_dict[key] = [points[i]]
    filtered_points = []
    for key, pts in voxel_dict.items():
        pts_array = np.array(pts)
        filtered_points.append(np.mean(pts_array, axis=0))
    return np.array(filtered_points)

def compute_normal_pca(points, k=10):
    """
    Compute surface normal vectors using Principal Component Analysis (PCA).
    
    For each point, this function:
    1. Finds k nearest neighbors
    2. Computes the covariance matrix of these neighbors
    3. Performs eigen decomposition
    4. Uses the eigenvector with smallest eigenvalue as the normal vector
    
    Args:
        points: Input point cloud (Nx3 array)
        k: Number of neighbors to consider for normal estimation
        
    Returns:
        Average normal vector for the point cloud, or None if computation fails
    """
    if len(points) < k:
        return None
    tree = KDTree(points)
    normals = []
    for p in points:
        k_eff = min(k, len(points))
        _, idxs = tree.query(p, k=k_eff)
        neighbors = points[idxs]
        center = np.mean(neighbors, axis=0)
        centered = neighbors - center
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # Eigenvector corresponding to smallest eigenvalue
        # Flip normal if it points towards the camera (positive Z)
        if normal[2] > 0:
            normal = -normal
        normals.append(normal)
    normals = np.array(normals)
    avg_normal = np.mean(normals, axis=0)
    norm_val = np.linalg.norm(avg_normal)
    if norm_val < 1e-10:
        return None
    return avg_normal / norm_val  # Return normalized average normal vector

#===============================================================================
# INITIALIZATION
#===============================================================================

#-------------------------------------------------------------------------------
# UI Setup - HSV Trackbars
#-------------------------------------------------------------------------------
# Create window and trackbars for HSV color filtering adjustment
cv2.namedWindow('HSV Trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('HSV Trackbars', 400, 300)
# First HSV range (for red color which wraps around the hue circle)
cv2.createTrackbar('Lower H1', 'HSV Trackbars', 0, 179, nothing)
cv2.createTrackbar('Lower S1', 'HSV Trackbars', 120, 255, nothing)
cv2.createTrackbar('Lower V1', 'HSV Trackbars', 50, 255, nothing)
cv2.createTrackbar('Upper H1', 'HSV Trackbars', 7, 179, nothing)
cv2.createTrackbar('Upper S1', 'HSV Trackbars', 255, 255, nothing)
cv2.createTrackbar('Upper V1', 'HSV Trackbars', 255, 255, nothing)
# Second HSV range (for red color which wraps around the hue circle)
cv2.createTrackbar('Lower H2', 'HSV Trackbars', 170, 179, nothing)
cv2.createTrackbar('Lower S2', 'HSV Trackbars', 120, 255, nothing)
cv2.createTrackbar('Lower V2', 'HSV Trackbars', 50, 255, nothing)
cv2.createTrackbar('Upper H2', 'HSV Trackbars', 180, 179, nothing)
cv2.createTrackbar('Upper S2', 'HSV Trackbars', 255, 255, nothing)
cv2.createTrackbar('Upper V2', 'HSV Trackbars', 255, 255, nothing)

#-------------------------------------------------------------------------------
# RealSense Pipeline Setup
#-------------------------------------------------------------------------------
pipe = rs.pipeline()  # Initialize RealSense pipeline
cfg = rs.config()  # Create a configuration object

# Path to the BAG file (recording) - update this with the actual file path
BAG_FILE_PATH = r"C:\Users\johnn\Desktop\component.bag"
# BAG_FILE_PATH = r"C:\Users\johnn\Desktop\compressor4.bag"

# Configure pipeline to use the BAG file instead of a live camera
cfg.enable_device_from_file(BAG_FILE_PATH)

# Start the pipeline with the configuration
profile = pipe.start(cfg)
playback = profile.get_device().as_playback()  # Get playback device for BAG file control

# Create alignment object to align depth frames to color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Get camera intrinsic parameters for 3D calculations
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_profile.get_intrinsics()
fx, fy = intr.fx, intr.fy  # Focal lengths
cx, cy = intr.ppx, intr.ppy  # Principal points

# Define Region of Interest (ROI) - center region of the frame
frame_width = 640
frame_height = 480
top_left = (frame_width // 4, frame_height // 4)
bottom_right = (3 * frame_width // 4, 3 * frame_height // 4)
roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv2.rectangle(roi_mask, top_left, bottom_right, 255, -1)

max_detection_depth = 0.5  # Maximum depth in meters to consider for detection
# Morphological operation kernels for noise removal
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Initialize RealSense post-processing filters
spatial = rs.spatial_filter()  # Spatial filter to smooth depth data
temporal = rs.temporal_filter()  # Temporal filter to smooth depth data over time

#-------------------------------------------------------------------------------
# Parameter Optimization Setup
#-------------------------------------------------------------------------------
# Centroid Stabilization Variables
num_centroids = 2  # Number of centroids to track
# Buffer to store recent centroid positions for each tracked point (for smoothing)
centroid_buffers = [deque(maxlen=5) for _ in range(num_centroids)]
# Store fixed centroids after stabilization
fixed_centroids = [None for _ in range(num_centroids)]

# sigma_values = [0.05, 0.1, 0.2, 0.3, 0.5, 1, 3, 5, 10]
# simplification_rates = [0.3, 0.5, 0.7, 0.9]

# Parameter values to test
sigma_values = [100]  # Sigma parameter for Gaussian smoothing
simplification_rates = [0.5]  # Simplification rate

# # Alternative parameters for different neighborhood sizes
# # For k=50 neighbors
# sigma_values = [5]
# simplification_rates = [0.5]

base_voxel_size = 0.0002  # Base voxel size in meters

# Print voxel lengths for each simplification rate
for rate in simplification_rates:
    voxel_length = base_voxel_size / rate
    print(f"Simplification Rate: {rate} -> Voxel Length: {voxel_length:.6f} m")

# Create all combinations of sigma and simplification rate values to test
combination_list = [(s, r) for s in sigma_values for r in simplification_rates]
total_combinations = len(combination_list)

#-------------------------------------------------------------------------------
# Testing Phase Variables
#-------------------------------------------------------------------------------
# Optimization Phase Variables
test_frames_per_combination = 500       # Number of frames to accumulate for each parameter combination
stabilization_frames = 10  # Initial frames to skip for stabilization
# Number of optimization rounds to run
NUM_OPTIMIZATION_ROUNDS = 1
optimization_round = 1
# List to store results of all optimization rounds
# Format: (round number, {(sigma, rate): (stabilization_ratio, frame_avg_list)})
optimization_results_list = []
# Current phase: 'optimization' -> parameter testing, 'sampling' -> final sampling phase
current_phase = 'optimization'

# Variables specific to optimization phase
test_frame_counter = 0
sigma_rate_index = 0
current_sigma, current_rate = combination_list[sigma_rate_index]
# Store results for each combination: (stabilization_ratio, list of frame errors)
combination_results = { comb: (0, []) for comb in combination_list }
current_combination_frame_avgs = []
convergence_reached = False  # True when moving average difference is below threshold

# Sampling Phase Variables
post_stabilization_frames = 10  # Frames to stabilize after optimization phase
post_stabilization_done = False
FRAMES_PER_SAMPLE = 1000        # Number of frames per sample
STABILIZATION_FRAMES_SAMPLE = 2  # Stabilization frames before each sample
NUM_SAMPLES = 1                 # Total number of samples to collect
sample_count = 0
sample_stage = 'idle'
sample_stabilization_frame_count = 0
sample_collecting_frame_count = 0
current_sample_frame_avgs = []   # Store average errors for each frame in current sample
all_frame_diffs_per_sample = []  # Store all frame differences for each sample
sample_within_counts = []        # Count of frames within threshold for each sample
sample_total_counts = []         # Total frames in each sample
sample_percentages = []          # Percentage of frames within threshold for each sample

# Target angle settings
desired_angle_diff = 45.0  # Target angle difference between normal vectors (degrees)
# desired_angle_diff = 66
# desired_angle_diff = 60

#-------------------------------------------------------------------------------
# Visualization Setup
#-------------------------------------------------------------------------------
# Real-time Plot Initialization
plt.ion()  # Enable interactive mode for real-time plotting
fig, ax = plt.subplots()
# ax.set_title("Normal Vector Angle Differences (Real-Time)")
ax.set_xlabel("Frame")
ax.set_ylabel("Frame Average Angle Difference (degrees)")
line_individual, = ax.plot([], [], 'b-', label='Frame Average')
ax.axhline(desired_angle_diff, color='r', linestyle='--', label='Desired Angle')
ax.axhline(desired_angle_diff + 2, color='g', linestyle='--', label='Desired +2°')
ax.axhline(desired_angle_diff - 2, color='g', linestyle='--', label='Desired -2°')
ax.legend()
scatter_plot = None
plt.show()

#-------------------------------------------------------------------------------
# Statistics Tracking Variables
#-------------------------------------------------------------------------------
# Point Count Tracking
total_points_before_voxel = 0  # Total points before voxel filtering (for statistics)
total_points_after_voxel = 0   # Total points after voxel filtering (for statistics)
frames_processed = 0
points_before_list = []
points_after_list = []

# Processing Control Variables
frame_count = 0
point_cloud_update_interval = 10  # Update point cloud every 10 frames
stored_points = None  # Point cloud to maintain for 10 frames
stored_normal = None  # Normal vector to maintain for 10 frames
previous_normals = [None] * num_centroids  # Store previous normal vectors for each centroid
angle_threshold = 0.2  # Angle change threshold (degrees) for stability detection
stabilization_frames = 10  # Initial frames to skip for stabilization
recording_started = False  # Flag to indicate if recording has started
angle_tolerance = 2  # Allowed tolerance from desired_angle_diff (degrees)
tolerance_frame_count = 0  # Count of consecutive frames within angle tolerance
required_tolerance_frames = 15  # Required consecutive frames within tolerance to start recording

#===============================================================================
# MAIN PROCESSING LOOP
#===============================================================================

try:
    # Main processing loop
    while True:
        frame_count += 1

        #-----------------------------------------------------------------------
        # Frame Acquisition
        #-----------------------------------------------------------------------
        # Read frames from the .bag file
        try:
            frames = pipe.wait_for_frames()
        except RuntimeError:
            print("Reached the end of the .bag file. No more frames available.")
            break

        # Align depth frame to color frame
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert frame data to numpy arrays
        color_image_rgb = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)

        #-----------------------------------------------------------------------
        # Status Display
        #-----------------------------------------------------------------------
        if frame_count < stabilization_frames:
            # Initial stabilization phase
            sigma_to_use = 0.000001
            simplification_rate_to_use = simplification_rates[0]
            status_text = f"[Frame {frame_count}][Initial Stabilizing]"
        else:
            if current_phase == 'optimization':
                # Optimization phase - testing different parameter combinations
                current_status = f"Optimization Round {optimization_round}: Testing (sigma = {current_sigma}, sim_rate = {current_rate})"
                status_text = f"[Frame {frame_count}][{current_status}]"
            elif current_phase == 'sampling':
                # Sampling phase - collecting data with optimal parameters
                if sample_stage == 'idle':
                    sample_stage = 'stabilizing'
                    sample_stabilization_frame_count = 0
                    print("\n[Sampling Phase Started] Collecting multiple samples with final selected parameters.\n")
                if sample_stage == 'stabilizing':
                    # Stabilizing before collecting sample
                    sample_stabilization_frame_count += 1
                    current_status = f"Sample Stabilizing ({sample_stabilization_frame_count}/{STABILIZATION_FRAMES_SAMPLE})"
                elif sample_stage == 'collecting':
                    # Actively collecting sample data
                    sample_collecting_frame_count += 1
                    current_status = f"Sample Collecting ({sample_collecting_frame_count}/{FRAMES_PER_SAMPLE})"
                else:
                    current_status = "Sampling Finished"
                status_text = f"[Frame {frame_count}][{current_status}]"
        
        # Display status text on the image
        cv2.putText(color_image, status_text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        #-----------------------------------------------------------------------
        # Image Processing and Centroid Detection
        #-----------------------------------------------------------------------
        # Apply spatial and temporal filters to the depth frame
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_image = np.asanyarray(depth_frame.get_data())
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Get current HSV trackbar values for red color filtering
        # First range (lower red hues)
        lower_h1 = cv2.getTrackbarPos('Lower H1', 'HSV Trackbars')
        lower_s1 = cv2.getTrackbarPos('Lower S1', 'HSV Trackbars')
        lower_v1 = cv2.getTrackbarPos('Lower V1', 'HSV Trackbars')
        upper_h1 = cv2.getTrackbarPos('Upper H1', 'HSV Trackbars')
        upper_s1 = cv2.getTrackbarPos('Upper S1', 'HSV Trackbars')
        upper_v1 = cv2.getTrackbarPos('Upper V1', 'HSV Trackbars')
        # Second range (higher red hues - wraps around hue circle)
        lower_h2 = cv2.getTrackbarPos('Lower H2', 'HSV Trackbars')
        lower_s2 = cv2.getTrackbarPos('Lower S2', 'HSV Trackbars')
        lower_v2 = cv2.getTrackbarPos('Lower V2', 'HSV Trackbars')
        upper_h2 = cv2.getTrackbarPos('Upper H2', 'HSV Trackbars')
        upper_s2 = cv2.getTrackbarPos('Upper S2', 'HSV Trackbars')
        upper_v2 = cv2.getTrackbarPos('Upper V2', 'HSV Trackbars')

        # Create HSV range arrays
        lower_red1 = np.array([lower_h1, lower_s1, lower_v1])
        upper_red1 = np.array([upper_h1, upper_s1, upper_v1])
        lower_red2 = np.array([lower_h2, lower_s2, lower_v2])
        upper_red2 = np.array([upper_h2, upper_s2, upper_v2])

        # Create binary masks for red color detection
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  # First range mask
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  # Second range mask
        red_mask = cv2.bitwise_or(mask1, mask2)  # Combine both range masks
        red_mask = cv2.bitwise_and(red_mask, roi_mask)  # Apply ROI limitation
        
        # Apply depth threshold limitation
        depth_meters = depth_image * 0.001  # Convert depth to meters
        valid_depth_mask = (depth_meters <= max_detection_depth).astype(np.uint8) * 255
        red_mask = cv2.bitwise_and(red_mask, valid_depth_mask)
        
        # Apply morphological operations to clean up the mask
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)  # Remove small noise
        red_mask = cv2.GaussianBlur(red_mask, (3,3), 0)  # Smooth the mask
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)  # Fill small holes

        # Find contours in the mask for object detection
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area, max_area = 5, 3000  # Filter contours by area
        if len(contours) > 0:
            # Sort contours by area (largest first)
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            # Take only the top few contours (num_centroids)
            selected_contours = contours_sorted[:num_centroids]
            detected_centroids = []
            for cnt in selected_contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                # Calculate centroid using moments
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    u = int(M["m10"] / M["m00"])  # x-coordinate of centroid
                    v = int(M["m01"] / M["m00"])  # y-coordinate of centroid
                    detected_centroids.append((u, v))
        else:
            detected_centroids = []

        #-----------------------------------------------------------------------
        # Centroid Stabilization
        #-----------------------------------------------------------------------
        # Average centroid positions over several frames to reduce jitter
        for i in range(num_centroids):
            if i < len(detected_centroids):
                # Add current detected centroid to the buffer
                centroid_buffers[i].append(detected_centroids[i])
            if len(centroid_buffers[i]) > 0:
                # Calculate average position from the buffer
                avg_u = int(np.mean([c[0] for c in centroid_buffers[i]]))
                avg_v = int(np.mean([c[1] for c in centroid_buffers[i]]))
                fixed_centroids[i] = (avg_u, avg_v)
            else:
                fixed_centroids[i] = None

        #-----------------------------------------------------------------------
        # Normal Vector Calculation
        #-----------------------------------------------------------------------
        frame_normals = []
        search_radius = 0.005  # 5mm radius around each centroid
        search_px = 50  # Search radius in pixels for better performance

        # Process each detected and stabilized centroid
        for idx, fc in enumerate(fixed_centroids):
            if fc is None:
                continue
            (u, v) = fc  # Get 2D image coordinates of the centroid
            
            # Get depth value at the centroid and convert to meters
            depth_val = depth_image[v, u] * 0.001
            if depth_val <= 0 or depth_val > max_detection_depth:
                continue
                
            # Draw the centroid on the image
            cv2.circle(color_image, (u, v), 5, (0,255,0), -1)
            text = f"Centroid {idx+1}: ({u},{v}), {depth_val:.3f}m"
            cv2.putText(color_image, text, (u+10, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        
            # Convert 2D image coordinates to 3D world coordinates
            Xc = (u - cx) * depth_val / fx  # X coordinate (right)
            Yc = (v - cy) * depth_val / fy  # Y coordinate (down)
            Zc = depth_val                  # Z coordinate (forward)

            # Define a search region around the centroid in the image
            umin = max(u - search_px, 0)
            umax = min(u + search_px, frame_width - 1)
            vmin = max(v - search_px, 0)
            vmax = min(v + search_px, frame_height - 1)
            
            # Extract depth values in the search region
            local_depth = depth_image[vmin:vmax+1, umin:umax+1]
            # Create coordinate grids for the search region
            grid_u, grid_v = np.meshgrid(np.arange(umin, umax+1),
                                         np.arange(vmin, vmax+1))
                                         
            # Filter points with valid depth values
            valid_local = (local_depth > 0)
            if not np.any(valid_local):
                continue
                
            # Extract valid points and convert to 3D coordinates
            local_z = local_depth[valid_local] * 0.001  # Convert to meters
            local_u = grid_u[valid_local]
            local_v = grid_v[valid_local]
            points = np.stack([
                (local_u - cx) * local_z / fx,  # X coordinates
                (local_v - cy) * local_z / fy,  # Y coordinates
                local_z                         # Z coordinates
            ], axis=-1)
            
            # Calculate 3D distance from each point to the centroid
            dist3D = np.sqrt((points[:,0] - Xc)**2 + (points[:,1] - Yc)**2 + (points[:,2] - Zc)**2)
            # Filter points within the search radius
            within_circle = dist3D < search_radius
            points_circle = points[within_circle]
            
            # Skip if not enough points for reliable normal calculation
            if len(points_circle) < 100:
                continue
                
            # Select parameters based on current phase
            if current_phase == 'optimization':
                # In optimization phase, use the current test parameters
                sigma_to_use = current_sigma
                simplification_rate_to_use = current_rate
            else:
                # In sampling phase, use the final selected parameters
                sigma_to_use = current_sigma
                simplification_rate_to_use = current_rate
                
            # Calculate voxel size based on base size and simplification rate
            voxel_size = base_voxel_size / simplification_rate_to_use

            # Point Cloud Processing
            # Step 1: Apply Gaussian smoothing to reduce noise
            points_circle_smoothed = gaussian_filter_point_cloud(points_circle, k=100, sigma=sigma_to_use)
            # Step 2: Apply voxel grid filter to reduce point density
            points_circle_smoothed = voxel_grid_filter(points_circle_smoothed, voxel_size=voxel_size)
            
            # Track point counts before and after processing for statistics
            points_before = points_circle.shape[0]
            points_after = points_circle_smoothed.shape[0]
            total_points_before_voxel += points_before
            total_points_after_voxel += points_after
            frames_processed += 1
            points_before_list.append(points_before)
            points_after_list.append(points_after)
            
            # Calculate normal vector using PCA method
            current_normal = compute_normal_pca(points_circle_smoothed, k=10)
            
            if current_normal is not None:
                # If recording hasn't started yet or this is the first frame
                if not recording_started:
                    normal_vec = current_normal
                # If recording has started
                else:
                    # Compare with previous normal vector to check stability
                    dot_val = np.clip(np.dot(current_normal, previous_normals[idx]), -1.0, 1.0)
                    angle_diff = np.degrees(np.arccos(dot_val))
                    
                    # If angle change is too large, keep the previous normal vector
                    if angle_diff > angle_threshold:
                        normal_vec = previous_normals[idx]
                        cv2.putText(color_image, f"Centroid {idx+1} Angle Change: {angle_diff:.2f}deg (Previous)", 
                                  (u, v + 60 + idx*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                    else:
                        # If angle change is small enough, update the normal vector
                        normal_vec = current_normal
                        previous_normals[idx] = normal_vec  # Update only if change is within threshold
                        cv2.putText(color_image, f"Centroid {idx+1} Angle Change: {angle_diff:.2f}deg", 
                                  (u, v + 60 + idx*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
                
                # Add the normal vector to the list for this frame
                frame_normals.append(normal_vec)

                # Visualization
                # Try to fit an ellipse to the projected points (better visualization than circle)
                points_2d = np.column_stack((local_u[within_circle], local_v[within_circle]))
                if len(points_2d) >= 5:  # Ellipse fitting requires at least 5 points
                    try:
                        ellipse = cv2.fitEllipse(points_2d.astype(np.float32))
                        # Draw the ellipse: (center, (major_axis, minor_axis), angle)
                        cv2.ellipse(color_image, ellipse, (0,255,255), 2)
                    except:
                        # Fall back to circle if ellipse fitting fails
                        circle_radius_px = int(round((search_radius * fx) / depth_val))
                        circle_radius_px = max(5, min(circle_radius_px, 200))
                        cv2.circle(color_image, (u, v), circle_radius_px, (0,255,255), 2)
                else:
                    # Not enough points for ellipse, draw circle instead
                    circle_radius_px = int(round((search_radius * fx) / depth_val))
                    circle_radius_px = max(5, min(circle_radius_px, 200))
                    cv2.circle(color_image, (u, v), circle_radius_px, (0,255,255), 2)
                
                # Draw the normal vector as an arrow
                scale = 50  # Scale factor for arrow length
                end_pt = (int(u + normal_vec[0]*scale),
                          int(v + normal_vec[1]*scale))
                cv2.arrowedLine(color_image, (u, v), end_pt, (255,0,0), 2, tipLength=0.3)
                
                # Display point statistics
                cv2.putText(color_image, f"Before Voxel: {points_before}", (u, v + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
                cv2.putText(color_image, f"After Voxel: {points_after}", (u, v + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

        #-----------------------------------------------------------------------
        # Parameter Optimization Phase
        #-----------------------------------------------------------------------
        if current_phase == 'optimization' and frame_count >= stabilization_frames:
            test_frame_counter += 1
            
            # Calculate average angle difference between normal vectors (if we have at least 2)
            if len(frame_normals) >= 2:
                differences = []
                # Calculate angles between all pairs of normal vectors
                for i in range(len(frame_normals)):
                    for j in range(i+1, len(frame_normals)):
                        # Calculate angle between two normal vectors using dot product
                        dot_val = np.clip(np.dot(frame_normals[i], frame_normals[j]), -1.0, 1.0)
                        angle = np.degrees(np.arccos(dot_val))
                        differences.append(angle)
                # Calculate average of all angle differences
                frame_avg = np.mean(differences) if differences else np.nan
            else:
                frame_avg = np.nan

            # Initialize list for the first frame of each parameter combination
            if test_frame_counter == 1:
                current_combination_frame_avgs = []
                convergence_reached = False

            # Add the current frame's average angle to the list
            current_combination_frame_avgs.append(frame_avg)
            
            # Check for convergence using moving average
            window_size = 50
            convergence_threshold = 0.1
            if len(current_combination_frame_avgs) >= 2 * window_size:
                # Compare average of recent window with previous window
                recent_window = current_combination_frame_avgs[-window_size:]
                previous_window = current_combination_frame_avgs[-2*window_size:-window_size]
                moving_avg_diff = abs(np.mean(recent_window) - np.mean(previous_window))
                
                # If change in moving average is small enough, consider it converged
                if moving_avg_diff < convergence_threshold:
                    print(f"[Convergence] sigma = {current_sigma}, rate = {current_rate} converged with moving average change {moving_avg_diff:.3f}° (below threshold {convergence_threshold}°).")
                    convergence_reached = True

            # Finish testing current parameter combination if:
            # 1. We've processed enough frames OR
            # 2. The angle measurements have converged
            if test_frame_counter >= test_frames_per_combination or convergence_reached:
                # Calculate stabilization ratio: percentage of frames within target range
                valid_avgs = [val for val in current_combination_frame_avgs if not np.isnan(val)]
                if valid_avgs:
                    # Count frames where angle is within ±2° of desired angle
                    count_within = sum(1 for val in valid_avgs if abs(val - desired_angle_diff) <= 2)
                    stabilization_ratio = (count_within / len(valid_avgs)) * 100
                else:
                    stabilization_ratio = 0
                    
                # Store results for this parameter combination
                combination_results[(current_sigma, current_rate)] = (stabilization_ratio, current_combination_frame_avgs.copy())
                
                # Move to next parameter combination
                sigma_rate_index += 1
                if sigma_rate_index < total_combinations:
                    # Update parameters to next combination
                    current_sigma, current_rate = combination_list[sigma_rate_index]
                    
                    # Reset BAG file to beginning to test next combination with same data
                    try:
                        playback.seek(datetime.timedelta(seconds=0))
                        print("[Optimization] .bag file reset complete.")
                    except Exception as e:
                        print(f"[Optimization] .bag file reset failed: {e}")
                        break
                        
                    # Reset counters and lists for next combination
                    test_frame_counter = 0
                    current_combination_frame_avgs = []
                    convergence_reached = False
                else:
                    # All combinations tested in this round -> store round results
                    optimization_results_list.append((optimization_round, combination_results.copy()))
                    print(f"\n[Optimization Round {optimization_round} Complete] Results saved for this round.\n")
                    
                    # Check if we need to run another optimization round
                    if optimization_round < NUM_OPTIMIZATION_ROUNDS:
                        optimization_round += 1
                        
                        # Reset everything for next round
                        test_frame_counter = 0
                        sigma_rate_index = 0
                        current_sigma, current_rate = combination_list[sigma_rate_index]
                        combination_results = {comb: (0, []) for comb in combination_list}
                        current_combination_frame_avgs = []
                        convergence_reached = False
                        
                        # Reset BAG file for next round
                        try:
                            playback.seek(datetime.timedelta(seconds=0))
                            print(f"[Optimization Round {optimization_round}] .bag file reset complete.")
                        except Exception as e:
                            print(f"[Optimization Round {optimization_round}] .bag file reset failed: {e}")
                            break
                        continue
                    else:
                        # All optimization rounds complete -> determine final best parameters
                        final_scores = {}
                        
                        # Aggregate results from all rounds
                        for round_info in optimization_results_list:
                            _, round_results = round_info
                            for comb, (ratio, _) in round_results.items():
                                if comb not in final_scores:
                                    final_scores[comb] = []
                                final_scores[comb].append(ratio)
                                
                        # Find combination with highest average stabilization ratio
                        final_best_comb, scores = max(final_scores.items(), key=lambda item: np.mean(item[1]))
                        final_avg_ratio = np.mean(scores)
                        
                        # Print final selected parameters
                        print("\n[Final Parameter Selection]")
                        print(f"Final selected parameters: sigma = {final_best_comb[0]}, simplification_rate = {final_best_comb[1]} (Average stabilization ratio: {final_avg_ratio:.2f}%)")
                        
                        # Set final parameters and transition to sampling phase
                        final_best_combination = final_best_comb
                        current_sigma, current_rate = final_best_combination
                        current_phase = 'sampling'
                        post_stabilization_done = False
                        post_stabilization_counter = 0
        
        #-----------------------------------------------------------------------
        # Post-Stabilization Phase
        #-----------------------------------------------------------------------
        if current_phase == 'sampling' and not post_stabilization_done:
            # Wait for a few frames after transitioning to sampling phase
            # to ensure system is stable with the final parameters
            if post_stabilization_counter < post_stabilization_frames:
                post_stabilization_counter += 1
            else:
                post_stabilization_done = True
                print(f"\n[Post Stabilization Completed] {post_stabilization_frames} frames stabilized.\n")
                
        #-----------------------------------------------------------------------
        # Sampling Phase
        #-----------------------------------------------------------------------
        if current_phase == 'sampling' and post_stabilization_done:
            # Exit loop if we've collected all required samples
            if sample_count >= NUM_SAMPLES:
                break

            # Calculate average angle difference between normal vectors in this frame
            if len(frame_normals) >= 2:
                differences = []
                # Calculate angles between all pairs of normal vectors
                for i in range(len(frame_normals)):
                    for j in range(i+1, len(frame_normals)):
                        dot_val = np.clip(np.dot(frame_normals[i], frame_normals[j]), -1.0, 1.0)
                        angle = np.degrees(np.arccos(dot_val))
                        differences.append(angle)
                frame_avg = np.mean(differences) if differences else np.nan
                
                # Check if angle is within desired range ± tolerance
                if not recording_started and not np.isnan(frame_avg):
                    if abs(frame_avg - desired_angle_diff) <= angle_tolerance:
                        # Increment counter when frame is within tolerance
                        tolerance_frame_count += 1
                        # Display stabilization progress
                        cv2.putText(color_image, f"Stabilizing: {tolerance_frame_count}/{required_tolerance_frames}", 
                                  (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                        
                        # Start recording if angle has been stable for enough frames
                        if tolerance_frame_count >= required_tolerance_frames:
                            recording_started = True
                            # Save normal vectors at recording start as reference
                            previous_normals = frame_normals.copy()
                            print(f"\n[Recording Started] Frame {frame_count}: Angle difference {frame_avg:.2f}° maintained within {desired_angle_diff}±{angle_tolerance}° for {required_tolerance_frames} frames\n")
                    else:
                        # Reset counter if angle falls outside tolerance
                        tolerance_frame_count = 0
                        cv2.putText(color_image, f"Waiting for stability: {frame_avg:.2f}°", 
                                  (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                
                # Store data only after recording has started
                if recording_started:
                    # Add current frame's average angle to the list
                    current_sample_frame_avgs.append(frame_avg)
                    
                    # Update real-time plot
                    x_data = np.arange(1, len(current_sample_frame_avgs) + 1)
                    y_data = np.array(current_sample_frame_avgs)
                    line_individual.set_xdata(x_data)
                    line_individual.set_ydata(y_data)
                    
                    # Color points based on whether they're within target range
                    if scatter_plot is None:
                        scatter_plot = ax.scatter(
                            x_data,
                            y_data,
                            c=['green' if (not np.isnan(val) and abs(val - desired_angle_diff) <= 2) else 'red' for val in y_data],
                            zorder=3
                        )
                    else:
                        scatter_plot.set_offsets(np.column_stack((x_data, y_data)))
                        scatter_plot.set_color(['green' if (not np.isnan(val) and abs(val - desired_angle_diff) <= 2) else 'red' for val in y_data])
                    
                    # Update plot limits and redraw
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                
                # Display current status on image
                status_color = (0,255,0) if recording_started else (0,255,255)
                status_text = f"Recording: {frame_avg:.2f}°" if recording_started else f"Waiting: {frame_avg:.2f}°"
                cv2.putText(color_image, status_text, (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            else:
                frame_avg = np.nan

            # Process sample data after collecting enough frames
            if sample_collecting_frame_count >= FRAMES_PER_SAMPLE or len(current_sample_frame_avgs) >= FRAMES_PER_SAMPLE:
                # Calculate statistics for this sample
                valid_avgs = [val for val in current_sample_frame_avgs if not np.isnan(val)]
                if valid_avgs:
                    # Count frames where angle is within ±2° of desired angle
                    count_within = sum(1 for val in valid_avgs if abs(val - desired_angle_diff) <= 2)
                    stabilization_ratio = (count_within / len(valid_avgs)) * 100
                    sample_avg_diff = np.mean(valid_avgs)
                    outlier_count = len(valid_avgs) - count_within
                else:
                    stabilization_ratio = 0
                    sample_avg_diff = np.nan
                    outlier_count = 0

                # Print sample statistics
                print(f"[Sample {sample_count+1}] Stabilization Ratio: {stabilization_ratio:.2f}% "
                      f"(Frames within threshold: {count_within}, Total valid frames: {len(valid_avgs)})")
                print(f"[Sample {sample_count+1}] Sample Average Angle Difference: {sample_avg_diff:.2f}°")
                print(f"[Sample {sample_count+1}] Outlier Count: {outlier_count}")

                # Create summary plot for this sample
                fig_sample, ax_sample = plt.subplots()
                ax_sample.plot(x_data, y_data, 'b-', label='Frame Average')
                ax_sample.scatter(
                    x_data,
                    y_data,
                    c=['green' if abs(val - desired_angle_diff) <= 2 else 'red' for val in y_data],
                    zorder=3
                )
                ax_sample.axhline(desired_angle_diff, color='r', linestyle='--', label='Desired Angle')
                ax_sample.axhline(desired_angle_diff + 2, color='g', linestyle='--', label='Desired +2°')
                ax_sample.axhline(desired_angle_diff - 2, color='g', linestyle='--', label='Desired -2°')
                # ax_sample.set_title(f"Sample Test (Frames: {len(current_sample_frame_avgs)})\n"
                #                     f"Avg Diff: {sample_avg_diff:.2f}°, Outliers: {outlier_count}")
                ax_sample.set_xlabel("Frame")
                ax_sample.set_ylabel("Average Angle Difference (degrees)")
                ax_sample.legend()
                plt.show()

                # Store results for this sample
                all_frame_diffs_per_sample.append(current_sample_frame_avgs.copy())
                sample_within_counts.append(count_within)
                sample_total_counts.append(len(valid_avgs))
                sample_percentages.append(stabilization_ratio)

                # Reset for next sample
                sample_count += 1
                current_sample_frame_avgs = []
                sample_collecting_frame_count = 0

                # Reset BAG file and filters for next sample
                try:
                    playback.seek(datetime.timedelta(seconds=0))
                    # Recreate filters to ensure clean state
                    spatial = rs.spatial_filter()
                    temporal = rs.temporal_filter()
                    print("Sampling phase: BAG file and filters reset complete.")
                except Exception as e:
                    print("Sampling phase: BAG file and filters reset failed:", e)

        #-----------------------------------------------------------------------
        # Display Output
        #-----------------------------------------------------------------------
        # Draw ROI boundary
        cv2.rectangle(color_image, top_left, bottom_right, (255,0,0), 2)
        
        # Display output images
        cv2.imshow("RGB with ROI", color_image)
        cv2.imshow("Red Mask", red_mask)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    # Handle any exceptions that occur during execution
    print(f"Error: {e}")

finally:
    # Clean up resources regardless of how the program exits
    pipe.stop()  # Stop the RealSense pipeline
    cv2.destroyAllWindows()  # Close all OpenCV windows
    plt.ioff()  # Turn off interactive plotting mode

    #---------------------------------------------------------------------------
    # Display Filtering Statistics
    #---------------------------------------------------------------------------
    # Calculate and print voxel filtering statistics
    if frames_processed > 0:
        avg_before = total_points_before_voxel / frames_processed
        avg_after = total_points_after_voxel / frames_processed
        reduction_rate = (1 - (avg_after / avg_before)) * 100 if avg_before > 0 else 0
        print("\n[Voxel Grid Filter Statistics]")
        print(f"  Processed Centroids: {frames_processed}")
        print(f"  Average points before filtering: {avg_before:.2f}")
        print(f"  Average points after filtering: {avg_after:.2f}")
        print(f"  Average reduction rate: {reduction_rate:.2f}%")
    else:
        print("\n[Voxel Grid Filter Statistics] No centroids were processed.")

    #---------------------------------------------------------------------------
    # Print Optimization Results
    #---------------------------------------------------------------------------
    # Print optimization test results summary
    if optimization_results_list:
        print("\n[Parameter Optimization Test Results Summary]")
        for round_info in optimization_results_list:
            round_num, round_results = round_info
            print(f"  Round {round_num}:")
            for comb, (ratio, frame_list) in round_results.items():
                print(f"    sigma = {comb[0]}, rate = {comb[1]}: Stabilization ratio = {ratio:.2f}% (Data points: {len(frame_list)})")
    else:
        print("\n[No optimization test results available.]")

    #---------------------------------------------------------------------------
    # Generate Summary Plots
    #---------------------------------------------------------------------------
    # Create plots showing angle differences for each sample
    if len(all_frame_diffs_per_sample) > 0:
        # Create a multi-panel figure with one subplot per sample
        fig_all, axs = plt.subplots(nrows=len(all_frame_diffs_per_sample),
                                    ncols=1, figsize=(8, 3 * len(all_frame_diffs_per_sample)))
        if len(all_frame_diffs_per_sample) == 1:
            axs = [axs]  # Ensure axs is a list when there's only one sample
            
        # Plot data for each sample
        for i, frame_diffs in enumerate(all_frame_diffs_per_sample):
            ax_i = axs[i]
            x_data = np.arange(1, len(frame_diffs) + 1)
            y_data = np.array(frame_diffs)
            # Color points based on whether they're within target range
            colors = ['green' if abs(y - desired_angle_diff) <= 2 else 'red' for y in y_data]
            ax_i.plot(x_data, y_data, 'b-', label='Frame Average')
            ax_i.scatter(x_data, y_data, c=colors, zorder=3)
            ax_i.set_xlabel("Frame")
            ax_i.set_ylabel("Angle error (degrees)")
            ax_i.grid(True)
            ax_i.axhline(desired_angle_diff, color='r', linestyle='--', label='Desired Angle')
            ax_i.axhline(desired_angle_diff + 2, color='g', linestyle='--', label='Desired +2°')
            ax_i.axhline(desired_angle_diff - 2, color='g', linestyle='--', label='Desired -2°')
            ax_i.legend()
            
        plt.tight_layout()
        # Save plot with timestamp in filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_all.savefig(f"frame_avg_{timestamp}.png")
        plt.show()

    # Create summary bar chart of stabilization percentages
    if sample_percentages:
        fig_summary, ax_summary = plt.subplots()
        sample_numbers = np.arange(1, len(sample_percentages)+1)
        ax_summary.bar(sample_numbers, sample_percentages, color='skyblue', edgecolor='black')
        # ax_summary.set_title("Percentage of Frames Within Desired Threshold per Sample")
        ax_summary.set_xlabel("Sample Number")
        ax_summary.set_ylabel("Percentage (%)")
        ax_summary.set_ylim(0, 100)
        # Add percentage labels above each bar
        for i, perc in enumerate(sample_percentages):
            ax_summary.text(i+1, perc + 2, f"{perc:.1f}%", ha='center', va='bottom')
        plt.tight_layout()
        # Save plot with timestamp in filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_summary.savefig(f"sample_summary_{timestamp}.png")
        plt.show()
    else:
        print("\n[Not enough sample data to generate summary plots.]")
