import os
import shutil

# Specify the directory containing the image dataset
input_dir = r'../../data/images/raw'  # Update this path to your dataset location

# Check if the directory exists
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"The directory {input_dir} does not exist. Please check the path.")

# Get a list of all .jpg files in the directory
file_list = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]

# Check if any files were found
if not file_list:
    raise ValueError(f"No .jpg files found in the directory: {input_dir}")

# Process each file
for old_name in file_list:
    # Check if the filename contains an underscore (to ensure it matches the expected format)
    if '_' in old_name:
        # Extract the timestamp portion (everything before the underscore)
        timestamp = old_name[:old_name.index('_')]
        
        # Construct the new filename by appending .jpg
        new_name = f"{timestamp}.jpg"
        
        # Define full paths for old and new filenames
        old_path = os.path.join(input_dir, old_name)
        new_path = os.path.join(input_dir, new_name)
        
        # Check if the new filename already exists to avoid overwriting
        if os.path.exists(new_path):
            print(f"Warning: File {new_name} already exists. Skipping rename to prevent overwrite.")
        else:
            # Perform the rename
            shutil.move(old_path, new_path)
            print(f"Renamed {old_name} to {new_name}")
    else:
        print(f"Skipping {old_name}: No underscore found, assuming correct format.")

print("Renaming process completed.")