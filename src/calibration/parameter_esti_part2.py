import cv2
import numpy as np
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
import glob
import os
import re

# Example function to load sun positions (replace with your actual data)
def get_sun_positions(image_paths, timestamps):
    location = EarthLocation(lat=37.0916*u.deg, lon=-2.3636*u.deg, height=490.587*u.m)
    sun_directions = []
    for ts in timestamps:
        time = Time(ts, scale='utc', location=location)
        sun = get_sun(time)
        altaz = sun.transform_to(AltAz(obstime=time, location=location))
        zenith = (90*u.deg - altaz.alt).value * np.pi / 180.0  # Radians
        azimuth = altaz.az.value * np.pi / 180.0  # Radians
        # Convert to 3D unit vector (ENU frame)
        X = np.cos(azimuth) * np.sin(zenith)
        Y = np.sin(azimuth) * np.sin(zenith)
        Z = np.cos(zenith)
        sun_directions.append([X, Y, Z])
    return np.array(sun_directions)

def extract_timestamp_and_sequence(image_path):
    # Get the filename from the path
    filename = os.path.basename(image_path)
    
    # Use regex to match pattern: digits{14}_digits{5}\.jpg
    # Assuming timestamp is 14 digits (YYYYMMDDHHMMSS) and sequence is 5 digits
    match = re.match(r'(\d{14})\.jpg$', filename)
    
    if match:
        timestamp = match.group(1)
        return timestamp
    else:
        return None, None
    
def convert_datetime_format(datetime_strings):
    formatted_dates = []
    for dt_str in datetime_strings:
        # Extract date (first 8 characters) and time (remaining characters)
        date_part = f"{dt_str[0:4]}-{dt_str[4:6]}-{dt_str[6:8]}"
        time_part = f"{dt_str[8:10]}:{dt_str[10:12]}:{dt_str[12:]}"
        # Combine with 'T' in between
        formatted_date = f"{date_part}T{time_part}"
        formatted_dates.append(formatted_date)
    return formatted_dates


# Load cropped images and detect sun positions
image_path = '../../data/processed/cropped/*.jpg'
image_paths = []
for item in glob.glob(image_path):
    image_paths.append(item)

input_datetime = []
for item in glob.glob(image_path):
    item = extract_timestamp_and_sequence(item)
    input_datetime.append(item)


timestamps = convert_datetime_format(input_datetime)

# print('image_paths: ', image_paths)
# print('\ninput_datetime: ', input_datetime)
# print('\ntimestamps: ', timestamps)


image_points = []  # (u, v) coordinates

for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # Detect bright sun
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            image_points.append([cX, cY])
    else:
        raise ValueError(f"No sun detected in {path}")

image_points = np.array(image_points)
world_directions = get_sun_positions(image_paths, timestamps)

# Normalize image points to homogeneous coordinates
image_points_hom = np.hstack((image_points, np.ones((image_points.shape[0], 1))))

# Form A matrix for DLT
A = []
for i in range(len(image_points)):
    x, y, w = image_points_hom[i]
    X, Y, Z = world_directions[i]
    A.append([0, -Z, Y, 0, Z*x, -Y*x, 0, -X, X*y, -Z*y, Y*y, -X*x])
    A.append([Z, 0, -X, -Z*x, 0, X*x, Z*y, -Y*y, 0, X*y, -Z*y, Y*x])
A = np.array(A)

# Solve using SVD
_, _, Vt = np.linalg.svd(A)
P = Vt[-1].reshape(3, 4)

# Decompose P into K and [R|t] using initial K from Part 1
# h, w, _ = cv2.imread(image_paths[0]).shape
h, w = cv2.imread(image_paths[0]).shape[:0:-1]
print('h, w: ', h, w)
cx = w // 2
cy = h // 2
# Use focal length from Part 1 (adjust based on cropped radius)
image_radius = min(w, h) // 2
g_o_90 = 2 * np.sin(90.0 * np.pi / 360.0)  # Recompute for consistency
fx = fy = image_radius / g_o_90
K_init = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

# Decompose projection matrix
H = P[:, :3]
U, S, Vt = np.linalg.svd(H)
R = np.dot(U, Vt)
if np.linalg.det(R) < 0:
    R = -R  # Ensure proper rotation matrix
t = P[:, 3] / S[0]  # Translation (negligible for infinity)

# Form the full extrinsic matrix [R|t]
extrinsic = np.hstack((R, t.reshape(3, 1)))

print("Initial Projection Matrix P:\n", P)
print("Initial Rotation Matrix R:\n", R)
print("Initial Translation Vector t:\n", t)
print("Initial Extrinsic Matrix [R|t]:\n", extrinsic)