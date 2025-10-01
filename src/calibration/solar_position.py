import cv2
import numpy as np
import os
from typing import Dict, Tuple, List
from datetime import datetime
import glob
import csv
from dataset_access import get_fisheye_image_paths

def create_verification_directory(base_path: str = "./../../Results/sun_detection_verification") -> str:

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    verification_dir = os.path.join(base_path, f"verification_{timestamp}")
    os.makedirs(verification_dir, exist_ok=True)
    print(f"Created verification directory: {verification_dir}")
    return verification_dir

def detect_sun_position_with_marking(image_path: str, output_dir: str) -> Tuple[int, int]:
    """
    Detects the sun position in a fisheye image and marks it visually for verification.
    
    Args:
        image_path (str): Path to the input image file.
        output_dir (str): Directory to save marked verification images.
    
    Returns:
        Tuple[int, int]: Detected sun coordinates (u_i, v_i) or (None, None) if not detected.
    """
    # Step 1: Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_img = img.copy()  # Preserve original for marking
    h, w = img.shape[:2]
    # print(f"Processing {image_path} (dimensions: {w}x{h})")
    
    # Step 2: Preprocess for robustness - Gaussian blurring and edge detection
    blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Reduce noise
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Apply Canny edge detection for boundary enhancement
    edges = cv2.Canny(gray, 50, 150)
    
    # Step 3: Thresholding (saturate pixels > 200 in luminance equivalent)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Optional: Morphological closing to connect edges and fill small gaps
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Step 4: Blob detection via contours (primary method)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"  No contours found in {image_path}")
        # Save image with "No Sun Detected" text
        cv2.putText(original_img, "NO SUN DETECTED", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        output_filename = os.path.basename(image_path).replace('.jpg', '_no_sun.jpg')
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, original_img)
        return None, None
    
    # Select the largest contour (assumed to be the sun)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Filter small blobs (sun should be reasonably sized, e.g., >100 pixels)
    if area < 100:
        print(f"  Largest blob too small (area: {area}) in {image_path}")
        cv2.putText(original_img, f"TOO SMALL: {area}px", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        output_filename = os.path.basename(image_path).replace('.jpg', '_too_small.jpg')
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, original_img)
        return None, None
    
    # Compute centroid (u_i, v_i) as pixel coordinates
    M = cv2.moments(largest_contour)
    if M['m00'] != 0:
        u_i = int(M['m10'] / M['m00'])  # x-coordinate
        v_i = int(M['m01'] / M['m00'])  # y-coordinate
    else:
        print(f"  No moments computed for {image_path}")
        cv2.putText(original_img, "NO MOMENTS", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        output_filename = os.path.basename(image_path).replace('.jpg', '_no_moments.jpg')
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, original_img)
        return None, None
    
    # Step 5: Optional validation with HoughCircles for circular blobs
    hough_refined = True
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=50)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Check if Hough circle aligns with centroid (within 10 pixels)
            if abs(x - u_i) < 10 and abs(y - v_i) < 10:
                u_i, v_i = x, y  # Refine with circle center
                hough_refined = True
                print(f"  Hough circle refined sun at ({u_i}, {v_i})")
                break
    
    # Step 6: Visual marking for verification
    # Draw circle around detected sun (green for blob, blue for Hough refinement)
    circle_color = (255, 0, 0) if hough_refined else (0, 255, 0)  # BGR
    circle_thickness = 3 if hough_refined else 2
    
    # # Draw main detection circle (radius based on blob size)
    detection_radius = int(np.sqrt(area / np.pi))  # Approximate radius
    cv2.circle(original_img, (u_i, v_i), detection_radius, circle_color, circle_thickness)
    
    # # Draw center point
    cv2.circle(original_img, (u_i, v_i), 5, (0, 0, 255), -1)  # Red center dot
    
    # # Add coordinate text label
    text = f"Sun: ({u_i}, {v_i})"
    text_color = (255, 255, 255) if hough_refined else (0, 255, 0)  # White or green
    cv2.putText(original_img, text, (u_i + 10, v_i - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # # Add area information
    # area_text = f"Area: {area:.0f}px²"
    # cv2.putText(original_img, area_text, (50, h - 50), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # # Add detection method indicator
    # method_text = "HOUGH" if hough_refined else "BLOB"
    # cv2.putText(original_img, f"Method: {method_text}", (50, 30), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    # # Generate output filename
    status = "hough" if hough_refined else "blob"
    output_filename = os.path.basename(image_path).replace('.jpg', f'_{status}_detected.jpg')
    output_path = os.path.join(output_dir, output_filename)
    
    # Save marked image
    cv2.imwrite(output_path, original_img)
    print(f"  Detected sun at ({u_i}, {v_i})")
    #  with area {area} - Saved: {output_filename}
    
    return u_i, v_i

def process_filtered_images_with_verification(image_paths: List[str], 
                                           output_dir: str = "./../../Results/sun_detection_verification") -> Dict[str, Tuple[int, int]]:
    """
    Processes filtered images with visual marking for verification.
    
    Args:
        image_paths (List[str]): List of paths to filtered images.
        output_dir (str): Base directory for verification images.
    
    Returns:
        Dict[str, Tuple[int, int]]: Dictionary of {image_path: (u_i, v_i)}.
    """
    # Create timestamped verification directory
    verification_dir = create_verification_directory(output_dir)
    
    sun_positions = {}
    successful_detections = 0
    
    for path in image_paths:
        u, v = detect_sun_position_with_marking(path, verification_dir)
        if u is not None:
            sun_positions[path] = (u, v)
            successful_detections += 1
    
    print(f"\n=== SUN DETECTION SUMMARY ===")
    print(f"Processed {len(image_paths)} images")
    print(f"Successful detections: {successful_detections}")
    print(f"Detection rate: {successful_detections/len(image_paths)*100:.1f}%")
    print(f"Verification images saved to: {verification_dir}")
    
    return sun_positions

def write_sun_positions_to_csv(sun_positions: Dict[str, Tuple[int, int]], csv_path: str = "./../../Results/sun_positions1.csv") -> None:
    """
    Writes the detected sun positions to a CSV file.
    
    Args:
        sun_positions (Dict[str, Tuple[int, int]]): Dictionary of {image_path: (u_i, v_i)}.
        csv_path (str): Path to the output CSV file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["Image Name", "u_i", "v_i"])
        
        for path, (u, v) in sun_positions.items():
            image_name = str(os.path.splitext(os.path.basename(path))[0]).strip()
            writer.writerow([image_name, u, v])
    
    print(f"Sun positions written to: {csv_path}")

# Example usage (replace with your filtered image paths)
if __name__ == "__main__":
    # Example filtered image paths (update this list with your actual paths)
    img_paths = './../../dataset/Cropped/*.jpg'
    base_path = '/storage2/CV_Irradiance/datasets/'
    image_paths = get_fisheye_image_paths(base_path)
    for path in image_paths:
        print(path)

    # filtered_paths = []
    # for item in glob.glob(img_paths):
    #     filtered_paths.append(item)
    # # print(filtered_paths)
    
    # # Process images with visual marking
    # positions = process_filtered_images_with_verification(filtered_paths)

    # # Write positions to CSV
    # write_sun_positions_to_csv(positions)
    
    # print("\n=== DETECTED SUN POSITIONS ===")
    # for path, (u, v) in positions.items():
    #     print(f"{os.path.basename(path)}: ({u}, {v})")
    
    # print(f"\nCheck the '{os.path.basename(create_verification_directory())}' folder for verification images!")