# API Documentation

## Core Modules

### Calibration Module (`src/calibration/`)

#### `solar_position.py`

**Functions:**

- `detect_sun_position_with_marking(image_path, output_dir)`
  - Detects sun position in fisheye image
  - **Parameters:**
    - `image_path`: Path to input image
    - `output_dir`: Directory for verification images
  - **Returns:** `(u_i, v_i)` pixel coordinates or `(None, None)`

- `process_filtered_images_with_verification(image_paths, output_dir)`
  - Batch process multiple images
  - **Parameters:**
    - `image_paths`: List of image file paths
    - `output_dir`: Output directory for results
  - **Returns:** Dictionary mapping filenames to coordinates

#### `para_est_DLT.py`

**Functions:**

- `dlt_estimation(image_points, world_points)`
  - Direct Linear Transform parameter estimation
  - **Parameters:**
    - `image_points`: List of 2D image coordinates
    - `world_points`: List of 3D world coordinates
  - **Returns:** Projection matrix P (3x4)

- `read_sun_positions(csv_path)`
  - Read detected sun positions from CSV
  - **Parameters:**
    - `csv_path`: Path to CSV file
  - **Returns:** Dictionary of image names to coordinates

#### `expected_solar_position.py`

**Functions:**

- `parse_timestamp(image_name)`
  - Extract timestamp from image filename
  - **Parameters:**
    - `image_name`: Image filename (format: YYYYMMDDHHMMSS)
  - **Returns:** ISO timestamp string

### Optical Flow Module (`src/optical_flow/`)

#### `opticalflow_cloudData1.py`

**Functions:**

- `calc_flow(args, model, image1, image2)`
  - Compute optical flow between image pair
  - **Parameters:**
    - `args`: Configuration arguments
    - `model`: RAFT model instance
    - `image1, image2`: Preprocessed image tensors
  - **Returns:** `(flow, info)` tensors

- `load_and_preprocess_image(image_path, device)`
  - Load and preprocess single image
  - **Parameters:**
    - `image_path`: Path to image file
    - `device`: PyTorch device ('cuda' or 'cpu')
  - **Returns:** Preprocessed tensor [1, C, H, W]

- `process_image_sequence(model, args, input_dir, output_dir, device)`
  - Process complete image sequence
  - **Parameters:**
    - `model`: RAFT model instance
    - `args`: Configuration arguments
    - `input_dir`: Directory containing images
    - `output_dir`: Output directory for flow files
    - `device`: Computation device

#### `visualize_flow.py`

**Functions:**

- `visualize_flows(input_dir, output_dir)`
  - Generate flow visualizations
  - **Parameters:**
    - `input_dir`: Directory containing .pth flow files
    - `output_dir`: Output directory for visualizations

- `create_color_bar(height, width, color_map)`
  - Create color bar for flow magnitude
  - **Parameters:**
    - `height, width`: Dimensions in pixels
    - `color_map`: Color mapping function
  - **Returns:** Color bar image array

### Preprocessing Module (`src/preprocessing/`)

#### `preprocessing.py`

**Functions:**

- File renaming utilities for timestamp standardization
- Batch processing of image datasets

### Utilities Module (`src/utils/`)

#### `datasets.py`

**Classes:**

- `FlowDataset(data.Dataset)`
  - PyTorch dataset for optical flow data
  - **Methods:**
    - `__init__(self, aug_params, sparse, dataset, ...)`
    - `fetch(self, index)`: Get data sample
    - `__len__(self)`: Dataset length

## Configuration

### `config/config.py`

**Constants:**

- `SITE_CONFIG`: Location and timezone information
- `DATA_CONFIG`: Processing parameters
- `CLEAR_FILTER`: Irradiance filtering thresholds
- `IMAGE_CONFIG`: Image processing settings
- `FLOW_CONFIG`: Optical flow parameters
- `PATHS`: File system paths

## Error Handling

Common exceptions and their meanings:

- `ValueError`: Invalid input parameters or data
- `FileNotFoundError`: Missing input files
- `RuntimeError`: CUDA/GPU related errors
- `IndexError`: Array access out of bounds

## Performance Notes

- Use GPU acceleration when available
- Batch process images for efficiency
- Monitor memory usage with large datasets
- Consider image resizing for faster processing

## Examples

### Basic Usage

```python
from src.calibration.solar_position import detect_sun_position_with_marking
from src.optical_flow.opticalflow_cloudData1 import load_and_preprocess_image

# Detect sun position
u, v = detect_sun_position_with_marking('image.jpg', 'output/')

# Load image for optical flow
image_tensor = load_and_preprocess_image('image.jpg', 'cuda')
```

### Batch Processing

```python
from src.calibration.solar_position import process_filtered_images_with_verification

image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = process_filtered_images_with_verification(image_paths, 'results/')
```