import csv
import numpy as np
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
import pandas as pd

# Location from dataset
location = EarthLocation(lat=37.0916*u.deg, lon=-2.3636*u.deg, height=490.587*u.m)

# Function to parse timestamp from image name (e.g., "20191001122230" -> "2019-10-01T12:22:30")
def parse_timestamp(image_name):
    year = image_name[0:4]
    month = image_name[4:6]
    day = image_name[6:8]
    hour = image_name[8:10]
    minute = image_name[10:12]
    second = image_name[12:14]
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}"

# Read CSV and compute solar positions
input_csv = "../../results/sun_positions1.csv"
output_csv = "../../results/solar_vectors.csv"

# df = pd.read_csv(output_csv)
# print(df.info())

with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    writer.writerow(['Image Name', 'Zenith (rad)', 'Azimuth (rad)', 'X', 'Y', 'Z'])
    
    for row in reader:
        image_name = row['Image Name']
        u_i = row['u_i']
        v_i = row['v_i']
        
        # Parse timestamp
        timestamp = parse_timestamp(image_name)
        time = Time(timestamp, scale='utc', location=location)
        print(timestamp, time)
        
        # Compute sun position
        sun = get_sun(time)
        altaz = sun.transform_to(AltAz(obstime=time, location=location))
        
        # Zenith and azimuth in radians
        zenith = (90 * u.deg - altaz.alt).to(u.radian).value
        azimuth = altaz.az.to(u.radian).value
        
        # 3D unit vector (ENU frame)
        X = np.cos(azimuth) * np.sin(zenith)
        Y = np.sin(azimuth) * np.sin(zenith)
        Z = np.cos(zenith)
        
        # Normalize (though already unit length)
        vector = np.array([X, Y, Z])
        vector /= np.linalg.norm(vector)
        
        # Write to output CSV
        writer.writerow([image_name, zenith, azimuth, vector[0], vector[1], vector[2]])
        
        print(f"Processed {image_name}: Zenith={zenith:.3f}, Azimuth={azimuth:.3f}, Vector={vector}")

print(f"\nSolar vectors saved to {output_csv}")