"""
Configuration file for CV-Solar-Irradiance project.
Contains all path configurations and constants used across the project.
"""

import os
from pathlib import Path

# Project root directory (two levels up from config/)
PROJECT_ROOT = Path(__file__).parent.parent

# === DIRECTORY PATHS ===
PATHS = {
    # Data directories
    'data_root': PROJECT_ROOT / 'data',
    'raw_images': PROJECT_ROOT / 'data' / 'images' / 'raw',
    'cropped_images': PROJECT_ROOT / 'data' / 'processed' / 'cropped',
    'raw_data': PROJECT_ROOT / 'data' / 'raw',
    'processed_data': PROJECT_ROOT / 'data' / 'processed',
    
    # Results directories
    'results_root': PROJECT_ROOT / 'results',
    'calibration_results': PROJECT_ROOT / 'results' / 'calibration',
    'optical_flow_results': PROJECT_ROOT / 'results' / 'optical_flow',
    'visualizations': PROJECT_ROOT / 'results' / 'visualizations',
    'sun_detection_verification': PROJECT_ROOT / 'results' / 'sun_detection_verification',
    
    # Source code directories
    'src_root': PROJECT_ROOT / 'src',
    'calibration_src': PROJECT_ROOT / 'src' / 'calibration',
    'optical_flow_src': PROJECT_ROOT / 'src' / 'optical_flow',
    'preprocessing_src': PROJECT_ROOT / 'src' / 'preprocessing',
    'utils_src': PROJECT_ROOT / 'src' / 'utils',
}

# === SITE CONFIGURATION ===
SITE_CONFIG = {
    'latitude': 37.0916,    # PSA Metas, Spain
    'longitude': -2.3636,
    'altitude': 490.587,    # meters
    'timezone': 'Europe/Madrid',
    'name': 'PSA Metas',
}

# === DATA PROCESSING PARAMETERS ===
DATA_CONFIG = {
    'sample_stride': 10,
    'max_images': 120,
    'elevation_limits_deg': (10, 80),
}

# === CLEAR SKY FILTERING ===
CLEAR_FILTER = {
    'DNI_min': 600,    # Minimum Direct Normal Irradiance
    'DHI_max': 150,    # Maximum Diffuse Horizontal Irradiance
    'GHI_min': 500,    # Minimum Global Horizontal Irradiance
}

# === IMAGE PROCESSING ===
IMAGE_CONFIG = {
    'circular_extraction': {
        'radius_factor': 0.48,  # Factor of minimum dimension
        'center_offset': (40, 0),  # (x, y) offset from center
    },
    'sun_detection': {
        'threshold': 200,
        'min_area': 100,
        'gaussian_blur': (5, 5),
        'canny_thresholds': (50, 150),
        'morph_kernel_size': (3, 3),
    },
}

# === OPTICAL FLOW PARAMETERS ===
FLOW_CONFIG = {
    'device': 'cuda',
    'scale': 0,  # Scale factor for RAFT
    'iters': 20,  # Number of iterations
}

# === CSV FILE MAPPINGS ===
CSV_FILES = {
    'sun_positions': 'sun_positions1.csv',
    'solar_vectors': 'solar_vectors.csv',
    'filtered_dni': 'filtered_DNI_data.csv',
    'psa_timeseries': 'PSA_timeSeries_data.csv',
    'psa_metas': 'PSA_timeSeries_Metas.csv',
}

def get_path(key: str) -> Path:
    """Get a configured path by key."""
    if key in PATHS:
        return PATHS[key]
    else:
        raise KeyError(f"Path '{key}' not found in configuration")

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_csv_path(csv_key: str, data_type: str = 'results') -> Path:
    """Get full path to a CSV file."""
    filename = CSV_FILES[csv_key]
    if data_type == 'results':
        return PATHS['results_root'] / filename
    elif data_type == 'raw':
        return PATHS['raw_data'] / filename
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

# Ensure essential directories exist when config is imported
if __name__ != '__main__':
    for path_key in ['results_root', 'calibration_results', 'optical_flow_results', 
                    'visualizations', 'sun_detection_verification']:
        ensure_dir(PATHS[path_key])