# Results Directory

This directory stores all processing outputs and results.

## Structure

- `calibration/`: Camera calibration results including parameter files and verification images
- `optical_flow/`: Optical flow computation outputs (.pth files)
- `visualizations/`: Flow visualization images and analysis plots

## Output Files

### Calibration Results
- `sun_positions.csv`: Detected sun positions in images
- `solar_vectors.csv`: Theoretical solar position vectors
- `camera_parameters.txt`: Estimated camera intrinsic/extrinsic parameters
- `verification_images/`: Sun detection verification images

### Optical Flow Results
- `*.pth`: PyTorch tensors containing 2-channel optical flow
- Flow format: [height, width, 2] where channels are (u, v) motion vectors

### Visualizations
- `flow_*.jpg`: Color-coded optical flow visualizations
- `magnitude_*.jpg`: Flow magnitude heatmaps
- Analysis plots and statistics