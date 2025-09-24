### Edited by Sanjula

import sys
sys.path.append('core')
import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

# import datasets
from raft import RAFT
# from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

# @torch.no_grad()
# def save_flow(path, args, model, image1, image2):
#     os.system(f"mkdir -p {path}")
#     H, W = image1.shape[2:]
#     flow, info = calc_flow(args, model, image1, image2)
#     # print(flow[0].shape)
#     torch.save(flow[0].cpu(), f"{path}flow2c.pth")
#     # flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
#     # cv2.imwrite(f"{path}flow.jpg", flow_vis)
#     # heatmap = get_heatmap(info, args)
#     # vis_heatmap(f"{path}heatmap.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())

# @torch.no_grad()
# def demo_custom(model, args, device=torch.device('cuda')):
#     image1 = cv2.imread("./custom_clouds/image1.jpg")
#     image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#     image2 = cv2.imread("./custom_clouds/image2.jpg")
#     image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
#     image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
#     image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
#     H, W = image1.shape[1:]
#     image1 = image1[None].to(device)
#     image2 = image2[None].to(device)
#     save_flow('./custom_clouds/', args, model, image1, image2)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
#     parser.add_argument('--path', help='checkpoint path', type=str, default=None)
#     parser.add_argument('--url', help='checkpoint url', type=str, default=None)
#     parser.add_argument('--device', help='inference device', type=str, default='cpu')
#     args = parse_args(parser)
#     if args.path is None and args.url is None:
#         raise ValueError("Either --path or --url must be provided")
#     if args.path is not None:
#         model = RAFT(args)
#         load_ckpt(model, args.path)
#     else:
#         model = RAFT.from_pretrained(args.url, args=args)
        
#     if args.device == 'cuda':
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')
#     model = model.to(device)
#     model.eval()
#     demo_custom(model, args, device=device)

# if __name__ == '__main__':
#     main()

def load_and_preprocess_image(image_path, device):
    """
    Loads an image from disk, converts it to RGB, and preprocesses it into a PyTorch tensor.

    :param image_path: Path to the image file.
    :param device: The PyTorch device (e.g., 'cuda' or 'cpu').
    :return: Preprocessed image tensor [1, C, H, W].
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) # Convert to C, H, W
    image = image[None].to(device) # Add batch dimension: B, C, H, W
    return image

@torch.no_grad()
def process_image_pair_and_save(output_dir, output_prefix, args, model, image1_tensor, image2_tensor):
    """
    Calculates 2-channel optical flow for a single pair of preprocessed image tensors
    and saves the result as a .pth file with a given prefix.

    :param output_dir: Directory to save the output files.
    :param output_prefix: A string prefix for the output filenames (e.g., "frame_0000").
    :param args: Argparse arguments.
    :param model: The loaded RAFT model.
    :param image1_tensor: The first preprocessed image tensor [1, C, H, W].
    :param image2_tensor: The second preprocessed image tensor [1, C, H, W].
    """
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    # print(f"Calculating flow for {output_prefix}...")
    flow, info = calc_flow(args, model, image1_tensor, image2_tensor)

    # Save the 2-channel flow tensor as .pth
    # flow[0] gets rid of the batch dimension, .cpu() moves it to CPU for saving
    # --- MODIFIED LINE FOR CUSTOM NAMING ---
    torch.save(flow[0].cpu(), os.path.join(output_dir, f"{output_prefix}.pth"))
    # print(f"Saved optical flow to {os.path.join(output_dir, f'{output_prefix}.pth')}")
    # print("-" * 30)


@torch.no_grad()
def process_image_sequence(model, args, input_dir, output_dir, device=torch.device('cuda')):
    """
    Processes a sequence of images in a directory to calculate optical flow
    between consecutive frames and saves the results.

    :param model: The loaded RAFT model.
    :param args: Argparse arguments.
    :param input_dir: Directory containing the input image sequence.
    :param output_dir: Directory to save output flow and heatmap files.
    :param device: The PyTorch device (e.g., 'cuda' or 'cpu').
    """
    # List and sort image files to ensure correct temporal order
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if len(image_files) < 2:
        print(f"Error: Not enough images in '{input_dir}' to form pairs. Need at least 2 images.")
        return

    print(f"Found {len(image_files)} images in '{input_dir}'. Starting batch processing...")

    # Load the first image outside the loop to start the pairing
    prev_image_path = os.path.join(input_dir, image_files[0])
    image1_tensor = load_and_preprocess_image(prev_image_path, device)

    for i in range(1, len(image_files)):
        curr_image_path = os.path.join(input_dir, image_files[i])
        image2_tensor = load_and_preprocess_image(curr_image_path, device)

        # --- MODIFIED LINE FOR CUSTOM NAMING ---
        # Generate a descriptive prefix for output files based on the FIRST image's filename
        prefix = f"{os.path.splitext(image_files[i-1])[0]}"
        
        process_image_pair_and_save(output_dir, prefix, args, model, image1_tensor, image2_tensor)
        if i%50 == 0: print(f"{i}/{len(image_files)} files completed")
        
        # The current image becomes the previous image for the next iteration
        image1_tensor = image2_tensor

    print(f"Batch processing complete. All results saved to '{output_dir}'.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--path', help='checkpoint path', type=str, default=None)
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')

    # file_names = ["09_Metas", "10_Metas", "11_Metas", "11_Kontas"]
    file_names = ["09_Metas", "10_Metas"]
    for file in file_names:
        path = f"/storage2/SEA_RAFT/Datasets/cloud_data/searaft_cloud/{file}"
        output_path = f"/storage2/SEA_RAFT/Datasets/cloud_data/searaft_cloud/optical_flow/SEA_RAFT/{file}"
        subfiles = sorted(os.listdir(path))
        for subfile in subfiles:
            subfile_path = f"{path}/{subfile}"
            print(subfile_path)
            
    
            # --- NEW ARGUMENTS FOR INPUT/OUTPUT DIRECTORIES ---
            parser.add_argument('--input_dir', help='Directory containing the sequence of input images.', type=str, default=subfile_path)
            parser.add_argument('--output_dir', help='Directory to save output optical flow (.pth) and visualizations (.jpg).', type=str, default=f"{output_path}/{subfile}")

            args = parse_args(parser)

            if args.path is None and args.url is None:
                raise ValueError("Either --path or --url must be provided for model loading.")
            
            # Ensure output directory exists before starting
            os.makedirs(args.output_dir, exist_ok=True)

            print(f"Loading model from: {args.path if args.path else args.url}")
            if args.path is not None:
                model = RAFT(args)
                load_ckpt(model, args.path)
            else:
                model = RAFT.from_pretrained(args.url, args=args)
                
            if args.device == 'cuda':
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            model = model.to(device)
            model.eval() # Set model to evaluation mode

            # --- CALL THE NEW BATCH PROCESSING FUNCTION ---
            process_image_sequence(model, args, args.input_dir, args.output_dir, device=device)

if __name__ == '__main__':
    main()
