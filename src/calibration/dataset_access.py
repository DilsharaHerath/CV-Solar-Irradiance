import os
import glob
import re

def get_fisheye_image_paths(base_path, months=None, days=None, hours=None):

    # Default values for months, days, and hours if not provided
    if months is None:
        months = ['09_Metas']
    if days is None:
        days = [f'{i:02d}' for i in range(1, 32)]  # Days 01 to 31
    if hours is None:
        hours = [f'{i:02d}' for i in range(24)]    # Hours 00 to 23

    # Supported image extensions
    image_extensions = ('.jpg')
    
    # List to store image paths
    image_paths = []
    
    # Iterate through the folder structure
    for month in months:
        month_path = os.path.join(base_path, month)
        if not os.path.isdir(month_path):
            print(f"Directory not found: {month_path}")
            continue
        
        for day in days:
            day_path = os.path.join(month_path, day)
            if not os.path.isdir(day_path):
                continue
                
            for hour in hours:
                hour_path = os.path.join(day_path, hour)
                if not os.path.isdir(hour_path):
                    continue
                    
                # List all files in the hour directory
                try:
                    for file_name in os.listdir(hour_path):
                        # Check if the file is an image
                        if file_name.lower().endswith(image_extensions):
                            full_path = os.path.join(hour_path, file_name)
                            if os.path.isfile(full_path):
                                image_paths.append(full_path)
                except Exception as e:
                    print(f"Error accessing {hour_path}: {e}")
    
    return image_paths


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

# # Example usage:
# if __name__ == "__main__":
#     base_path = '/storage2/CV_Irradiance/datasets/'
#     image_paths = get_fisheye_image_paths(base_path)
#     for path in image_paths:
#         print(path)