# CV-Solar-Irradiance: Computer Vision for Solar Irradiance Prediction

A comprehensive computer vision system for solar irradiance prediction using fisheye sky imagery, optical flow analysis, and camera calibration techniques. This project combines traditional camera calibration methods with modern deep learning approaches to track cloud motion and predict solar irradiance changes.

## 🌞 Overview

This repository implements a complete pipeline for solar forecasting using:
- **Fisheye camera calibration** using solar position tracking
- **Optical flow computation** with RAFT (Recurrent All-Pairs Field Transforms)
- **Cloud motion analysis** for irradiance prediction
- **Integration with meteorological data** from PSA (Plataforma Solar de Almería)

## 📁 Project Structure

```
CV-Solar-Irradiance/
├── src/                          # Source code
│   ├── calibration/              # Camera calibration modules
│   │   ├── solar_position.py     # Sun detection in fisheye images
│   │   ├── para_est_DLT.py       # DLT parameter estimation
│   │   ├── expected_solar_position.py  # Theoretical solar positions
│   │   ├── calibrate_sky.py      # Sky calibration with MESOR data
│   │   ├── transformation.py     # Coordinate transformations
│   │   ├── CircularArea_extract.py  # Fisheye region extraction
│   │   └── ...                   # Other calibration utilities
│   ├── optical_flow/             # Optical flow processing
│   │   ├── opticalflow_cloudData1.py  # Main RAFT optical flow
│   │   └── visualize_flow.py     # Flow visualization tools
│   ├── preprocessing/            # Data preprocessing
│   │   └── preprocessing.py      # Image file preprocessing
│   └── utils/                    # Utility functions
│       └── datasets.py           # PyTorch dataset classes
├── data/                         # Data directory
│   ├── raw/                      # Raw data files
│   │   ├── filtered_DNI_data.csv # Direct Normal Irradiance data
│   │   ├── PSA_timeSeries_data.csv  # PSA meteorological data
│   │   └── ...                   # Other CSV files
│   └── processed/                # Processed data outputs
├── results/                      # Output results
├── config/                       # Configuration files
├── docs/                         # Documentation
├── scripts/                      # Execution scripts
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
# Core dependencies
pip install numpy pandas scipy opencv-python
pip install torch torchvision
pip install astropy pvlib-python
pip install matplotlib seaborn

# For RAFT model
pip install tensorboard
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/DilsharaHerath/CV-Solar-Irradiance.git
cd CV-Solar-Irradiance
```

2. Set up your data directories:
```bash
mkdir -p data/images/raw
mkdir -p data/processed/cropped
mkdir -p results/{calibration,optical_flow,visualizations}
```

3. Download or prepare your fisheye sky image dataset

## 📊 Dataset Information

- **Location**: PSA Metas station, Spain
  - Latitude: 37.0916°N
  - Longitude: -2.3636°E  
  - Altitude: 490.587m
- **Time Period**: September-November 2019
- **Data Types**:
  - Fisheye sky images with timestamps (format: `YYYYMMDDHHMMSS_00160.jpg`)
  - Irradiance measurements (GHI, DNI, DHI)
  - Temperature and pressure data
  - Clear sky filtering based on irradiance thresholds

## 🔧 Usage

### 1. Data Preprocessing

First, standardize your image filenames:
```bash
python src/preprocessing/preprocessing.py
```

### 2. Extract Fisheye Regions

Extract circular regions from rectangular fisheye images:
```bash
python src/calibration/CircularArea_extract.py
```

### 3. Camera Calibration

#### Detect Sun Positions
```bash
python src/calibration/solar_position.py
```

#### Calculate Theoretical Solar Positions
```bash
python src/calibration/expected_solar_position.py
```

#### Estimate Camera Parameters
```bash
python src/calibration/para_est_DLT.py
```

### 4. Optical Flow Analysis

Compute optical flow between consecutive images:
```bash
# Note: Requires RAFT model setup and core dependencies
python src/optical_flow/opticalflow_cloudData1.py \
    --cfg config/raft_config.yaml \
    --input_dir data/images/raw \
    --output_dir results/optical_flow
```

### 5. Visualization

Generate flow visualizations:
```bash
python src/optical_flow/visualize_flow.py \
    --input_dir results/optical_flow \
    --output_dir results/visualizations
```

## 📈 Key Features

### Camera Calibration
- **Solar Position Detection**: Automatic sun detection using blob analysis and Hough transforms
- **DLT Parameter Estimation**: Direct Linear Transform for camera parameter estimation
- **Fisheye Distortion Handling**: Specialized processing for fisheye lens distortion
- **Temporal Validation**: Cross-validation with astronomical calculations

### Optical Flow
- **RAFT Integration**: State-of-the-art optical flow estimation
- **Cloud Motion Tracking**: Specialized for meteorological applications  
- **Batch Processing**: Efficient processing of image sequences
- **GPU Acceleration**: CUDA support for faster computation

### Data Integration
- **MESOR Compatibility**: Integration with MESOR irradiance format
- **Clear Sky Filtering**: Automatic selection of clear sky conditions
- **Timezone Handling**: Proper UTC/local time conversions
- **Quality Control**: Data validation and filtering

## 🔬 Technical Details

### Calibration Pipeline
1. **Sun Detection**: Identify sun position in fisheye images using computer vision
2. **Solar Calculation**: Compute theoretical sun positions using astronomical libraries
3. **Parameter Estimation**: Use DLT to estimate camera intrinsic and extrinsic parameters
4. **Validation**: Cross-validate results with independent measurements

### Optical Flow Pipeline
1. **Preprocessing**: Normalize and prepare image pairs
2. **Flow Computation**: Use RAFT model to compute dense optical flow
3. **Post-processing**: Filter and validate flow vectors
4. **Visualization**: Generate interpretable flow visualizations

## 📝 Configuration

### Clear Sky Filtering Parameters
```python
CLEAR_FILTER = {
    'DNI_min': 600,    # Minimum Direct Normal Irradiance
    'DHI_max': 150,    # Maximum Diffuse Horizontal Irradiance  
    'GHI_min': 500     # Minimum Global Horizontal Irradiance
}
```

### Camera Parameters
- **Location**: PSA Metas, Spain (37.0916°N, -2.3636°E)
- **Timezone**: Europe/Madrid (UTC+1)
- **Elevation Limits**: 10-80 degrees (avoid low/high sun angles)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 References

- **RAFT**: Teed, Z. and Deng, J. "RAFT: Recurrent All-pairs Field Transforms for Optical Flow"
- **PVLib**: Holmgren, W. et al. "pvlib python: a python package for modeling solar energy systems"
- **PSA Dataset**: Plataforma Solar de Almería meteorological data

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **DilsharaHerath** - *Initial work* - [DilsharaHerath](https://github.com/DilsharaHerath)

## 🙏 Acknowledgments

- PSA (Plataforma Solar de Almería) for providing the dataset
- RAFT authors for the optical flow implementation
- PVLib community for solar position calculations
- OpenCV community for computer vision tools

---

**Note**: This project is part of ongoing research in solar forecasting and computer vision applications for renewable energy systems.