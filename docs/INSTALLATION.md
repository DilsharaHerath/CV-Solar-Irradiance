# Installation Guide

## System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optical flow processing)
- At least 8GB RAM
- Storage space for datasets (can be large)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DilsharaHerath/CV-Solar-Irradiance.git
cd CV-Solar-Irradiance
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n solar-cv python=3.9
conda activate solar-cv

# OR using venv
python -m venv solar-cv
source solar-cv/bin/activate  # Linux/Mac
# solar-cv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# For GPU support (optional but recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify Installation

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import astropy; print('Astropy installed successfully')"
```

### 5. Set Up Directories

```bash
# Create data directories
mkdir -p data/images/{raw,cropped}
mkdir -p results/{calibration,optical_flow,visualizations}

# Download RAFT model (if using pretrained)
# wget https://path-to-raft-model.pth -O models/raft-things.pth
```

### 6. Configure Paths

Edit `config/config.py` to match your setup:

```python
PATHS = {
    'raw_images': '/path/to/your/raw/images',
    'cropped_images': '/path/to/cropped/images',
    # ... other paths
}
```

## Troubleshooting

### Common Issues

1. **CUDA not available**
   ```bash
   # Install CUDA-compatible PyTorch
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **OpenCV issues**
   ```bash
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```

3. **Astropy installation problems**
   ```bash
   conda install astropy -c conda-forge
   ```

### Performance Optimization

- Use SSD storage for image datasets
- Ensure sufficient RAM (16GB+ recommended)
- Use GPU for optical flow computation
- Consider parallel processing for large datasets

## Next Steps

1. Prepare your fisheye image dataset
2. Update configuration files
3. Run the preprocessing pipeline
4. Execute camera calibration
5. Compute optical flow

See the main README.md for usage instructions.