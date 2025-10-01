# Configuration for CV-Solar-Irradiance

## Site Configuration
SITE_CONFIG = {
    'latitude': 37.0916,      # degrees North
    'longitude': -2.3636,     # degrees East  
    'altitude': 490.587,      # meters
    'timezone': 'Europe/Madrid',
    'station_name': 'PSA_Metas'
}

## Data Processing Configuration
DATA_CONFIG = {
    'sample_stride': 10,      # Use every Nth file when scanning
    'max_images': 120,        # Cap on number of frames to use
    'elevation_limits': [10, 80],  # Degrees, ignore very low/high elevations
}

## Clear Sky Filtering Thresholds
CLEAR_FILTER = {
    'DNI_min': 600,          # W/m² - Minimum Direct Normal Irradiance
    'DHI_max': 150,          # W/m² - Maximum Diffuse Horizontal Irradiance  
    'GHI_min': 500,          # W/m² - Minimum Global Horizontal Irradiance
}

## Image Processing Configuration
IMAGE_CONFIG = {
    'fisheye_radius_factor': 0.48,  # Fraction of image size for circular extraction
    'sun_detection_threshold': 200,  # Brightness threshold for sun detection
    'min_sun_area': 100,            # Minimum sun blob area in pixels
}

## Optical Flow Configuration
FLOW_CONFIG = {
    'device': 'cuda',        # 'cuda' or 'cpu'
    'batch_size': 1,         # Batch size for processing
    'scale': 0,              # RAFT scale parameter
    'iters': 12,             # RAFT iterations
}

## File Paths (relative to project root)
PATHS = {
    'raw_images': 'data/images/raw',
    'cropped_images': 'data/images/cropped', 
    'results': 'results',
    'calibration_results': 'results/calibration',
    'flow_results': 'results/optical_flow',
    'visualizations': 'results/visualizations',
}