import csv
import numpy as np
from typing import Dict, Tuple, List
import cv2
import matplotlib.pyplot as plt



# Read CSV files
def read_sun_positions(csv_path: str) -> Dict[str, Tuple[int, int]]:
    """Read detected sun positions from CSV."""
    positions = {}
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            positions[row['Image Name']] = (int(row['u_i']), int(row['v_i']))
    return positions

def read_solar_vectors(csv_path: str) -> Dict[str, np.ndarray]:
    """Read solar direction vectors from CSV."""
    vectors = {}
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            vectors[row['Image Name']] = np.array([float(row['X']), float(row['Y']), float(row['Z'])])
    return vectors

# DLT Implementation with improved normalization
def dlt_estimation(image_points: List[np.ndarray], world_points: List[np.ndarray]) -> np.ndarray:
    """
    Perform Direct Linear Transform to estimate projection matrix P.
    
    Args:
        image_points (List[np.ndarray]): List of 2D homogeneous points [u, v, 1].
        world_points (List[np.ndarray]): List of 3D homogeneous points [X, Y, Z, 0].
    
    Returns:
        np.ndarray: 3x4 projection matrix P.
    """
    A = []
    for x, X in zip(image_points, world_points):
        u, v, w = x
        X, Y, Z, W = X  # W=0 for points at infinity
        A.append([0, 0, 0, 0, -W*X, W*Y, -W*Z, -v*X, v*Y, -v*Z, 0, u*W])
        A.append([W*X, -W*Y, W*Z, 0, 0, 0, 0, u*X, -u*Y, u*Z, -v*W, 0])
    A = np.array(A)
    
    # Check condition number for ill-conditioning
    U, S, Vt = np.linalg.svd(A @ A.T)
    cond_number = S[0] / S[-1] if S[-1] != 0 else float('inf')
    print(f"Condition number of A^T A: {cond_number:.2e}")
    if cond_number > 1e6:
        print("Warning: Matrix A is ill-conditioned. Consider more diverse points.")
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    p = Vt[-1]  # Unnormalized solution
    P = p.reshape(3, 4)
    
    # Normalize by the largest absolute element to avoid zero division
    norm_factor = np.max(np.abs(P))
    if norm_factor == 0:
        raise ValueError("Projection matrix P is all zeros, indicating a degenerate solution.")
    P = P / norm_factor
    print(f"Raw P before normalization:\n{P * norm_factor}")
    print(f"Normalized P:\n{P}")
    
    return P

# Main execution
if __name__ == "__main__":
    # File paths
    # sun_positions_csv = "./../../Results/sun_positions1.csv"
    # solar_vectors_csv = "./../../Results/solar_vectors.csv"
    
    # # Read data
    # sun_positions = read_sun_positions(sun_positions_csv)
    # solar_vectors = read_solar_vectors(solar_vectors_csv)
    
    # # Common image names
    # common_images = set(sun_positions.keys()) & set(solar_vectors.keys())
    # if len(common_images) < 6:
    #     raise ValueError(f"Insufficient correspondences: {len(common_images)} < 6")
    
    # # Prepare lists for DLT (use a subset for debugging if needed)
    # image_points = [np.array([u, v, 1]) for img, (u, v) in sun_positions.items() if img in common_images]
    # world_points = [np.append(solar_vectors[img], 0) for img in common_images]  # [X, Y, Z, 0]
    
    # # Compute initial projection matrix
    # try:
    #     P = dlt_estimation(image_points, world_points)
    #     print("Initial Projection Matrix P:\n", P)
    # except np.linalg.LinAlgError as e:
    #     print(f"SVD failed: {e}. Check for degenerate correspondences or ill-conditioned matrix.")
    #     exit(1)
    
    # # Initial intrinsic matrix K (approximate for Q25, 1536x1536 cropped)
    # # h, w = 1536, 1536  # Adjust based on your cropped size
    # img_path = './../../dataset/Cropped/20191001122230.jpg'
    # # Initial intrinsic matrix K (approximate for Q25, 1536x1536 cropped)
    # img = cv2.imread(img_path)
    # h, w = img.shape[:2]
    # fx = fy = 685  # From Q25 focal length (1.6 mm) and pixel pitch
    # cx, cy = w // 2, h // 2
    # K_init = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # # Decompose P into [R | t]
    # H = P[:, :3]
    # try:
    #     U, S, Vt = np.linalg.svd(H)
    #     R = np.dot(U, Vt)
    #     if np.linalg.det(R) < 0:
    #         R = -R  # Ensure proper rotation
    #     t = P[:, 3] / S[0]  # Scale translation (negligible for infinity)
    # except np.linalg.LinAlgError as e:
    #     print(f"SVD decomposition failed: {e}. H matrix may be singular.")
    #     exit(1)
    
    # # Form extrinsic matrix
    # extrinsic = np.hstack((R, t.reshape(3, 1)))
    
    # print("Initial Rotation Matrix R:\n", R)
    # print("Initial Translation Vector t:\n", t)
    # print("Initial Extrinsic Matrix [R | t]:\n", extrinsic)
    
    # # Optional: Validate with reprojection
    # for i, (x, X) in enumerate(zip(image_points, world_points)):
    #     proj = P @ X
    #     if proj[2] != 0:  # Avoid division by zero
    #         proj = proj / proj[2]  # Homogeneous to Cartesian
    #         error = np.linalg.norm(proj[:2] - x[:2])
    #         print(f"Image {list(common_images)[i][:15]}... Reprojection Error: {error:.2f} pixels")

    u_vals, v_vals = zip(*[pos for pos in read_sun_positions("../../results/sun_positions1.csv").values()])
    plt.scatter(u_vals, v_vals)
    plt.title("Detected Sun Positions")
    plt.xlabel("u_i (pixels)")
    plt.ylabel("v_i (pixels)")
    plt.show()