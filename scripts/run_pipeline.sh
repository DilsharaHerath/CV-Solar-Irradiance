#!/bin/bash

# CV-Solar-Irradiance: Complete Processing Pipeline
# This script runs the full pipeline from raw images to optical flow analysis

set -e  # Exit on any error

echo "=== CV-Solar-Irradiance Processing Pipeline ==="
echo "Starting at: $(date)"

# Configuration
PROJECT_ROOT="/path/to/CV-Solar-Irradiance"  # Update this path
PYTHON_ENV="python"  # or "conda run -n myenv python"

# Change to project directory
cd "$PROJECT_ROOT"

echo ""
echo "Step 1: Data Preprocessing"
echo "=========================="
$PYTHON_ENV src/preprocessing/preprocessing.py
echo "✓ Image preprocessing completed"

echo ""
echo "Step 2: Extract Fisheye Regions"
echo "==============================="
$PYTHON_ENV src/calibration/CircularArea_extract.py
echo "✓ Circular region extraction completed"

echo ""
echo "Step 3: Camera Calibration"
echo "=========================="

echo "3a: Detecting sun positions..."
$PYTHON_ENV src/calibration/solar_position.py
echo "✓ Sun position detection completed"

echo "3b: Computing theoretical solar positions..."
$PYTHON_ENV src/calibration/expected_solar_position.py
echo "✓ Solar position calculation completed"

echo "3c: Estimating camera parameters..."
$PYTHON_ENV src/calibration/para_est_DLT.py
echo "✓ Camera parameter estimation completed"

echo ""
echo "Step 4: Optical Flow Analysis"
echo "============================="
$PYTHON_ENV src/optical_flow/opticalflow_cloudData1.py \
    --cfg config/raft_config.yaml \
    --input_dir data/images/cropped \
    --output_dir results/optical_flow \
    --device cuda
echo "✓ Optical flow computation completed"

echo ""
echo "Step 5: Flow Visualization"
echo "========================="
$PYTHON_ENV src/optical_flow/visualize_flow.py \
    --input_dir results/optical_flow \
    --output_dir results/visualizations
echo "✓ Flow visualization completed"

echo ""
echo "=== Pipeline Completed Successfully ==="
echo "Finished at: $(date)"
echo ""
echo "Results available in:"
echo "  - Calibration: results/calibration/"
echo "  - Optical Flow: results/optical_flow/"
echo "  - Visualizations: results/visualizations/"