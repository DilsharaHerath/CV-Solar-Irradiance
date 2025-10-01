# Data Directory

This directory contains the project datasets.

## Structure

- `raw/`: Raw data files including CSV irradiance data and original images
- `processed/`: Processed data outputs including calibration results and flow data

## Usage

Place your fisheye sky images in `raw/images/` and irradiance CSV files in `raw/`.
Processed outputs will be automatically saved to `processed/` by the pipeline.

## Data Format

### Images
- Format: JPEG
- Naming: `YYYYMMDDHHMMSS_00160.jpg` (timestamp format)
- Type: Fisheye sky images

### Irradiance Data
- Format: CSV with columns: Date-Time, GHI, DNI, DHI, Temp, Pressure
- Temporal resolution: 1 minute
- Units: W/m² for irradiance, °C for temperature, hPa for pressure