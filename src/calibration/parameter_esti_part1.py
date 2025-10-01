import cv2
import numpy as np

# Load the cropped image
img = cv2.imread('../../data/processed/cropped/20191001122230.jpg')
if img is None:
    raise ValueError("Image not loaded. Check the file path.")

# Get image dimensions (assuming cropped to a square or near-square)
h, w, _ = img.shape  # Height, Width (e.g., 1536x1536 if square cropped)
print(f"Image dimensions: {w}x{h}")
# print(img.shape[1])

# Assume the sky dome edge is the image boundary due to cropping
# Radius is half the minimum dimension (for a circular dome)
image_radius = min(w, h) // 2 - 30 # e.g., 768 if 1536x1536     I have taken 30 px gap
print(f"Estimated Image Radius: {image_radius}")

# Equisolid angle projection model: g_o(theta) = 2 * sin(theta/2)
theta_90_deg = 90.0 * np.pi / 180.0  # Convert to radians
g_o_90 = 2 * np.sin(theta_90_deg / 2)  # Approximately 1.414

# Estimate focal length: f_x ≈ f_y = image_radius / g_o(90°)
fx = fy = image_radius / g_o_90
print(f"Estimated Focal Length (fx, fy): {fx}")

# Principal point at image center
cx = w // 2  # e.g., 768 if 1536x1536
cy = h // 2  # e.g., 768 if 1536x1536
print(f"Principal Point (cx, cy): ({cx}, {cy})")

# Initial intrinsic matrix K
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float64)

print(f"Estimated Intrinsic Matrix K:\n{K}")

# Optional: Visualize the center for verification
cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
cv2.imwrite('center_verified1.jpg', img)