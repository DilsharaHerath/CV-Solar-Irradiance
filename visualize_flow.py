import sys
sys.path.append('core')

import torch
import numpy as np
import cv2
import os
import argparse # Ensure argparse is imported for main function
import subprocess

from utils.flow_viz import flow_to_image

# --- Functions copied from your provided snippet for color bar ---
def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        # Ensure images have same number of channels before concatenation
        # If image is grayscale and color_bar is BGR, convert image to BGR
        if len(image.shape) == 2: # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1: # Single channel image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Ensure both are BGR (3 channels)
        if image.shape[2] != 3:
            raise ValueError("Image must have 3 channels for vertical concatenation with color bar.")

        return cv2.vconcat([image, color_bar])
    else:
        # For horizontal, images must have same height
        # This function assumes image and color_bar are already compatible (e.g., both BGR)
        return cv2.hconcat([image, color_bar])
# ----------------------------------------------------------------

# def visualize_flows(input_dir, output_dir):
#     """
#     Loads 2-channel optical flow .pth files from an input directory,
#     visualizes them with a color bar indicating magnitude, and saves
#     the resulting images to an output directory.

#     :param input_dir: Path to the directory containing .pth flow files.
#     :param output_dir: Path to the directory where visualized flow images will be saved.
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     flow_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.pth')])

#     if not flow_files:
#         print(f"No .pth flow files found in '{input_dir}'.")
#         return

#     print(f"Found {len(flow_files)} .pth flow files in '{input_dir}'. Starting visualization...")

#     for flow_file_name in flow_files:
#         flow_path = os.path.join(input_dir, flow_file_name)
        
#         try:
#             # Load the 2-channel flow tensor
#             # It's expected to be [2, H, W] after loading from .pth
#             flow_tensor = torch.load(flow_path)
#             print(flow_tensor.size)
#             flow_vis = flow_to_image(flow_tensor[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
#             # cv2.imwrite(f"{path}flow.jpg", flow_vis)
    
#             # Permute to (H, W, 2) for flow_to_image function
#             # flow_data_hw2 = flow_tensor.permute(1, 2, 0).cpu().numpy()

#             # # Visualize the flow and get the magnitude image
#             # flow_color_image, flow_magnitude_image = flow_to_image(flow_data_hw2)

#             # # Create a color bar for magnitude
#             # # The width of the color bar should match the width of the flow image
#             # img_height, img_width = flow_color_image.shape[:2]
#             # color_bar_height = 50 # You can adjust this height
#             # magnitude_color_bar = create_color_bar(color_bar_height, img_width, cv2.COLORMAP_JET)

#             # # Add the color bar to the flow visualization
#             # combined_image = add_color_bar_to_image(flow_color_image, magnitude_color_bar, orientation='vertical')

#             # Define output filename (e.g., original_filename.jpg)
#             output_file_name = f"{os.path.splitext(flow_file_name)[0]}.jpg"
#             output_path = os.path.join(output_dir, output_file_name)

#             # Save the combined image
#             cv2.imwrite(output_path, flow_vis)
#             print(f"Visualized '{flow_file_name}' with color bar and saved to '{output_path}'")

#         except Exception as e:
#             print(f"Error processing '{flow_file_name}': {e}")
#             continue

#     print("Flow visualization complete.")

def visualize_flows(input_dir, output_dir):
    """
    Loads 2-channel optical flow .pth files from an input directory,
    visualizes them using flow_vis.flow_to_image (Baker et al. color wheel)
    and adds a magnitude color bar that aligns with flow_vis's internal normalization,
    then saves the resulting images to an output directory.

    :param input_dir: Path to the directory containing .pth flow files.
    :param output_dir: Path to the directory where visualized flow images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    flow_files = sorted(list_onedrive_files(input_dir))
    
    # flow_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.pth')])

    if not flow_files:
        print(f"No .pth flow files found in '{input_dir}'.")
        return

    print(f"Found {len(flow_files)} .pth flow files in '{input_dir}'. Starting visualization...")

    for flow_file_name in flow_files:
        flow_path = os.path.join(input_dir, flow_file_name)
        
        try:
            # Load the 2-channel flow tensor
            # It's expected to be [2, H, W] after loading from .pth
            flow_tensor = torch.load(flow_path)

            # Permute to (H, W, 2) for flow_vis.flow_to_image function
            flow_data_hw2 = flow_tensor.permute(1, 2, 0).cpu().numpy()

            # Calculate raw magnitude for this specific flow field
            

            height, width = flow_data_hw2.shape[:2]
            circle_radius_factor = 0.952
        # Define circle parameters: center at image center, radius as fraction of min dimension
            center = (width // 2, height // 2)
            radius = int(min(width, height)// 2 * circle_radius_factor)
            
            # Create a black mask with the same size as the image
            mask = np.zeros((height, width), dtype=np.uint8)
            # Draw a filled white circle on the mask
            cv2.circle(mask, center, radius, 255, -1)
            # Apply the mask to each channel of the image
            flow_data_hw2 = cv2.bitwise_and(flow_data_hw2, flow_data_hw2, mask=mask)

            u = flow_data_hw2[:, :, 0]
            v = flow_data_hw2[:, :, 1]

            raw_magnitude = np.sqrt(np.square(u) + np.square(v))
            
            # Determine the maximum magnitude that flow_vis.flow_to_image uses for its normalization.
            # This is the 'rad_max' value internally calculated by flow_vis.flow_to_image.
            # Add a small epsilon to prevent division by zero if all flow values are zero.
            flow_vis_normalization_max = np.max(raw_magnitude) + 1e-5 

            # Visualize the flow using the imported flow_vis.flow_to_image
            # Set convert_to_bgr=True to ensure OpenCV-compatible BGR output
            flow_color_image = flow_to_image(flow_data_hw2, convert_to_bgr=True) 

            # Create a color bar for magnitude that aligns with flow_vis's internal scaling.
            # We normalize the raw magnitude by the same max value that flow_vis uses,
            # then scale it to the 0-255 range for the color bar image.
            normalized_magnitude_for_colorbar = (raw_magnitude / flow_vis_normalization_max * 255)
            normalized_magnitude_for_colorbar = np.clip(normalized_magnitude_for_colorbar, 0, 255).astype(np.uint8)
            
            img_height, img_width = flow_color_image.shape[:2]
            color_bar_height = 50 # You can adjust this height
            
            # Create a color bar using the JET colormap for magnitude.
            # The gradient itself goes from 0 to 255. The interpretation is that
            # 255 on the color bar corresponds to 'flow_vis_normalization_max' in the flow field.
            magnitude_color_bar = create_color_bar(color_bar_height, img_width, cv2.COLORMAP_JET)

            # Add the color bar to the flow visualization
            combined_image = add_color_bar_to_image(flow_color_image, magnitude_color_bar, orientation='vertical')

            # Define output filename (e.g., original_filename.jpg)
            output_file_name = f"{os.path.splitext(flow_file_name)[0]}.jpg"
            output_path = os.path.join(output_dir, output_file_name)

            # Save the combined image
            cv2.imwrite(output_path, combined_image)
            print(f"Visualized '{flow_file_name}' with color bar and saved to '{output_path}'")

        except Exception as e:
            print(f"Error processing '{flow_file_name}': {e}")
            continue

    print("Flow visualization complete.")

def list_onedrive_files(path):
    """
    List .pth files in the specified OneDrive remote path and return their filenames.
    
    :param path: OneDrive remote path (e.g., 'onedrive:/content/Metas_09_01/01')
    :return: List of .pth filenames (e.g., '20190901180600_00160.pth')
    """
    try:
        result = subprocess.run(
            ["rclone", "ls", path],
            capture_output=True,
            text=True,
            check=True
        )
        files = result.stdout.strip().split("\n")
        pth_filenames = []
        for line in files:
            if line.strip() and line.strip().endswith('.pth'):
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    size, name = parts
                    pth_filenames.append(name)
        return sorted(pth_filenames)
    except subprocess.CalledProcessError as e:
        print(f"Error listing files: {e.stderr}")
        return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize 2-channel optical flow .pth files with magnitude color bar.")
    parser.add_argument('--input_dir', type=str, default='onedrive:/content/Metas_09_01/01',
                        help='OneDrive remote path containing .pth files.')
    parser.add_argument('--output_dir', type=str, default='onedrive:/content/Metas_09_01/visualizations',
                        help='OneDrive remote path to upload visualized .jpg files.')
    
    args = parser.parse_args()

    visualize_flows(args.input_dir, args.output_dir)
