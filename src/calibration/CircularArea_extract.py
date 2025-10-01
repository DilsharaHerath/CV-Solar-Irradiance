import cv2
import numpy as np
import os
from dataset_access import get_fisheye_image_paths, extract_timestamp_and_sequence

def extract_circular_region(image_path, output_path, center=None, radius=None):

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")

    height, width = img.shape[:2]
    
    if center is None:
        center = (width // 2 + 40, height // 2)
    if radius is None:
        radius = min(width, height) * 0.48

    radius = int(radius)
    if radius <= 0 or radius > min(width, height) // 2:
        raise ValueError(f"Invalid radius: {radius}. Must be positive and not exceed half the minimum dimension.")

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, (255), thickness=-1)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    x_center, y_center = center
    x_min = max(0, x_center - radius)
    x_max = min(width, x_center + radius)
    y_min = max(0, y_center - radius)
    y_max = min(height, y_center + radius)
    
    side_length = min(x_max - x_min, y_max - y_min)
    x_max = x_min + side_length
    y_max = y_min + side_length
    
    cropped_img = masked_img[y_min:y_max, x_min:x_max]
    image_name = extract_timestamp_and_sequence(image_path)
    
    filename = os.path.join(os.path.dirname(output_path), f"{image_name}.jpg")

    cv2.imwrite(filename, cropped_img)
    print(f"Cropped: {filename}")

if __name__ == "__main__":
    # Example usage: extract circular regions from fisheye images
    # Update these paths to match your data structure
    
    output_image_path = '../../data/processed/cropped/'
    os.makedirs(output_image_path, exist_ok=True)

    base_path = '../../data/images/raw/'
    image_paths = get_fisheye_image_paths(base_path)
    for path in image_paths:
        extract_circular_region(path, output_image_path)
    