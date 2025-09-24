import cv2
import numpy as np
import glob
import re
import os

def extract_circular_region(image_path, output_path, center=None, radius=None):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")

    # Get image dimensions
    height, width = img.shape[:2]
    
    # Automatically determine center and radius if not provided
    if center is None:
        center = (width // 2, height // 2)
    if radius is None:
        # Set radius to approximately 40% of the minimum dimension to capture the circular area
        radius = min(width, height) * 0.48

    # Create a mask with the same dimensions as the input image
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw a white circle on the mask
    cv2.circle(mask, center, int(radius), (255), thickness=-1)
    
    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    # Save the resulting image
    cv2.imwrite(output_path, masked_img)
    print(f"Processed image saved as {output_path}")

def extract_timestamp_and_sequence(image_path):
    """
    Extracts the timestamp and sequence number from an image path.
    
    Args:
        image_path (str): The full path to the image file.
    
    Returns:
        tuple: A tuple containing (timestamp, sequence) as strings.
               Returns (None, None) if extraction fails.
    """
    # Get the filename from the path
    filename = os.path.basename(image_path)
    
    # Use regex to match pattern: digits{14}_digits{5}\.jpg
    # Assuming timestamp is 14 digits (YYYYMMDDHHMMSS) and sequence is 5 digits
    match = re.match(r'(\d{14})_(\d{5})\.jpg$', filename)
    
    if match:
        timestamp = match.group(1)
        sequence = match.group(2)
        return timestamp
    else:
        return None, None

# Usage
input_image_path = './../../dataset/Calibration/*.jpg'  # Replace with your image file path
output_image_path = './../../dataset/Cropped'  # Output file path

# for img_path in glob.glob(input_image_path):
#     part = extract_timestamp_and_sequence(img_path)
#     output_path = output_image_path + f'/{part}.jpg'
#     # print(output_path)
#     extract_circular_region(img_path, output_path)
#     print(f'Completed {img_path}')