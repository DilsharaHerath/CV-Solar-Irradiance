"""
Fisheye self-calibration from sun track using MESOR irradiance file.

Requires: numpy, pandas, scipy, opencv-python, pvlib
pip install numpy pandas scipy opencv-python pvlib
"""

import os, re, math
import numpy as np
import pandas as pd
import cv2 as cv
from glob import glob
from datetime import datetime
from scipy.optimize import least_squares
import pvlib
from io import StringIO
import datetime as dt
from pathlib import Path

# ------------------------ CONFIG ------------------------
IMG_GLOB = "../../data/images/raw/*.jpg"          # e.g. "/data/DLR/*.jpg"
MESOR_PATH = "../../data/raw/PSA_timeSeries_Metas.csv"
LOCAL_TZ = "Europe/Madrid"                  # or "Etc/GMT-1" to force UTC+1 all season
SAMPLE_STRIDE = 10                          # use every Nth file when scanning (speeds up)
MAX_IMAGES = 120                          # cap used frames
CLEAR_FILTER = dict(DNI_min=600, DHI_max=150, GHI_min=500)
ELEV_LIMITS_DEG = (10, 80)                  # ignore very low/high elevations
# --------------------------------------------------------

# ---------- parsing helpers ----------
TS_REGEX = re.compile(r'(?P<ts>\d{14})_00160(?:\D|$)')

def ts_from_filename(fname: str, tz_str: str = "UTC+1"):
    tz = _fixed_offset_tz(tz_str)
    base = os.path.basename(fname)
    stem = os.path.splitext(base)[0]
    # Expect "YYYYMMDDhhmmss_00160"
    ts_part = stem.split("_")[0]
    dt_naive = dt.datetime.strptime(ts_part, "%Y%m%d%H%M%S")
    return dt_naive.replace(tzinfo=tz)

def parse_header_value(lines, key):
    for ln in lines:
        if ln.startswith("#") and key in ln:
            return float(ln.split(":")[1].strip())
    return None

def _parse_meta(lines):
    meta = {}
    for ln in lines:
        if ln.startswith("#location.latitude"):
            meta["lat"] = float(ln.split(":")[1].strip())
        elif ln.startswith("#location.longitude"):
            meta["lon"] = float(ln.split(":")[1].strip())
        elif ln.startswith("#location.altitude"):
            meta["alt_m"] = float(ln.split(":")[1].strip())
        elif ln.startswith("#timezone"):
            meta["tz_str"] = ln.split(" ", 1)[1].strip()  # e.g., "UTC+1"
    return meta

def _fixed_offset_tz(tz_str):
    # tz_str like "UTC+1" or "UTC-2"
    m = re.fullmatch(r"UTC([+-]\d{1,2})", tz_str)
    if not m:
        # Fallback: no offset
        return dt.timezone.utc
    hours = int(m.group(1))
    return dt.timezone(dt.timedelta(hours=hours), name=tz_str)

def attach_mesor_to_images(image_paths, df_mesor, tz_str):
    ts_img = pd.to_datetime(
        [ts_from_filename(p, tz_str) for p in image_paths]
    )
    img_df = pd.DataFrame({"img_path": image_paths, "ts_img": ts_img}).sort_values("ts_img")

    # nearest merge with 30 s tolerance
    merged = pd.merge_asof(
        img_df, 
        df_mesor.sort_index().reset_index().rename(columns={"datetime":"ts_mes"}),
        left_on="ts_img", right_on="ts_mes",
        direction="nearest", tolerance=pd.Timedelta("30s")
    )
    return merged

def read_mesor(mesor_path: str):
    mesor_path = Path(mesor_path)
    # read header for metadata
    header_lines = []
    with mesor_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("#"):
                break
            header_lines.append(line.strip())
    meta = _parse_meta(header_lines)

    # Primary: tab-separated with combined timestamp (your file)
    colnames6 = ["datetime", "GHI", "DNI", "DHI", "temp_C", "pressure_mbar"]
    df = pd.read_csv(
        mesor_path,
        comment="#",
        sep="\t",
        header=None,
        names=colnames6,
        engine="python"
    )

    # If it actually came as 7 columns (date + time split), fix up
    if df.shape[1] == 7:
        df.columns = ["date", "time", "GHI", "DNI", "DHI", "temp_C", "pressure_mbar"]
        df["datetime"] = df["date"].astype(str) + " " + df["time"].astype(str)
        df = df.drop(columns=["date", "time"])

    # Parse timestamp with the SPACE between date and time
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S", errors="raise")

    # Localize to the file's fixed offset (UTC+1 in your case)
    tz = _fixed_offset_tz(meta.get("tz_str", "UTC+0"))
    df["datetime"] = df["datetime"].dt.tz_localize(tz)

    # Set index for easy joins with images later
    df = df.set_index("datetime").sort_index()

    return df, meta

def parse_timestamp_from_filename(fname, tz_str):
    """Parse 'YYYYMMDDhhmmss_00160.jpg' -> tz-aware UTC Timestamp."""
    base = os.path.basename(fname)
    m = TS_REGEX.search(base)
    if not m:
        # optional fallback: accept any leading 14 digits if needed
        m = re.search(r'(\d{14})', base)
        if not m:
            raise ValueError(f"No timestamp found in {base}")
        ts = m.group(1)
    else:
        ts = m.group('ts')

    # interpret as LOCAL time first, then convert to UTC to match MESOR alignment
    local = pd.Timestamp(datetime.strptime(ts, "%Y%m%d%H%M%S")).tz_localize(tz_str)
    return local.tz_convert("UTC")

# ---------- vision helpers ----------
def detect_circle_center_radius(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 5)
    H, W = gray.shape
    circles = cv.HoughCircles(
        blur, cv.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=100, param2=50,
        minRadius=int(min(H,W)*0.35), maxRadius=int(min(H,W)*0.60)
    )
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        cx, cy, R = min(circles, key=lambda c: (c[0]-W/2)**2 + (c[1]-H/2)**2)
        return float(cx), float(cy), float(R)
    # fallback: fit circle to darkest rim
    mask = gray < np.percentile(gray, 5)
    ys, xs = np.nonzero(mask)
    x = xs.astype(np.float64); y = ys.astype(np.float64)
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    R = math.sqrt(sol[2] + cx**2 + cy**2)
    return float(cx), float(cy), float(R)

def find_sun_centroid(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    H,S,V = cv.split(hsv)
    mask = ((V > 240) & (S < 80)).astype(np.uint8) * 255
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        y, x = np.unravel_index(np.argmax(V), V.shape)
        return float(x), float(y)
    c = max(cnts, key=cv.contourArea)
    M = cv.moments(c)
    if M["m00"] == 0:
        x,y,w,h = cv.boundingRect(c)
        return float(x+w/2), float(y+h/2)
    return float(M["m10"]/M["m00"]), float(M["m01"]/M["m00"])

# ---------- geometry ----------
def rot_from_yaw_pitch_roll(yaw, pitch, roll):
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    return Rz @ Ry @ Rx

def unit_vector_from_az_el(az_deg, el_deg):
    az = np.deg2rad(az_deg); el = np.deg2rad(el_deg)
    cosel = np.cos(el)
    x = cosel * np.sin(az)   # East
    y = cosel * np.cos(az)   # North
    z = np.sin(el)           # Up
    return np.stack([x,y,z], axis=-1)

def kb_r_from_theta(theta, f, k1,k2,k3,k4):
    th2 = theta*theta
    return f*(theta + k1*th2*theta + k2*th2*th2*theta + k3*th2*th2*th2*theta + k4*th2*th2*th2*th2*theta)

def project_dir_to_px(v_cam, cx, cy, f, k1,k2,k3,k4):
    v = v_cam / np.linalg.norm(v_cam, axis=-1, keepdims=True)
    theta = np.arccos(np.clip(v[...,2], -1, 1))
    phi = np.arctan2(v[...,1], v[...,0])
    r = kb_r_from_theta(theta, f, k1,k2,k3,k4)
    u = cx + r*np.cos(phi); v_ = cy + r*np.sin(phi)
    return np.stack([u,v_], axis=-1)

# ---------- solar position via pvlib ----------
def solar_position(times_utc, lat, lon, alt, tempC, pressure_Pa):
    
    idx = pd.DatetimeIndex(times_utc)
    sp = pvlib.solarposition.get_solarposition(
        idx, latitude=lat, longitude=lon, altitude=alt,
        temperature=tempC, pressure=pressure_Pa,
        method='nrel_numpy'
    )
    return sp['azimuth'].to_numpy(), sp['apparent_elevation'].to_numpy()

# ---------- main calibration ----------
def calibrate_with_mesor(images_glob, mesor_path, tz_str, sample_stride, max_images):
    # --- read MESOR ---
    df, meta = read_mesor(mesor_path)
    lat, lon, alt = meta["lat"], meta["lon"], meta["alt"]

    # --- scan images & collect candidates ---
    files = sorted(glob(images_glob))[::max(1, sample_stride)]
    if not files:
        raise RuntimeError("No images matched IMG_GLOB")

    obs_uv = []
    times = []
    weights = []
    shape = None
    cx0 = cy0 = R0 = None

    for fp in files:
        try:
            t_utc = parse_timestamp_from_filename(fp, tz_str)
        except Exception:
            continue
        # nearest MESOR row within +/-60s
        try:
            near = df.index.get_indexer([t_utc], method="nearest", tolerance=pd.Timedelta("60s"))[0]
        except Exception:
            continue
        if near == -1:
            continue
        row = df.iloc[near]

        # clear-sky filters
        if (row["DNI"] < CLEAR_FILTER["DNI_min"]) or (row["DHI"] > CLEAR_FILTER["DHI_max"]) or (row["GHI"] < CLEAR_FILTER["GHI_min"]):
            continue

        # compute sun elev to prefilter (will refine later)
        # Quick estimate without temp/press is fine for gate — we’ll recompute below anyway.
        # (We could skip this and rely on final az/el.)
        # keep candidate; we’ll load and detect sun now
        img = cv.imread(fp, cv.IMREAD_COLOR)
        if img is None:
            continue

        if shape is None:
            shape = img.shape[:2]
            cx0, cy0, R0 = detect_circle_center_radius(img)

        sx, sy = find_sun_centroid(img)
        obs_uv.append([sx, sy])
        times.append(t_utc)
        # weight by DNI (normalize later)
        weights.append(float(row["DNI"]))
        if len(obs_uv) >= max_images:
            break

    if len(obs_uv) < 80:
        raise RuntimeError(f"Too few usable frames after filtering: {len(obs_uv)}")

    obs_uv = np.array(obs_uv, dtype=np.float64)
    times = pd.DatetimeIndex(times)
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.max(w)

    # --- precise solar positions with refraction corrections from MESOR ---
    # Build aligned arrays of temp & pressure for the selected timestamps
    sel_rows = df.reindex(times, method="nearest", tolerance=pd.Timedelta("60s"))
    az, el = solar_position(
        times.tz_localize("UTC"),
        lat, lon, alt,
        tempC=sel_rows["tempC"].to_numpy(),
        pressure_Pa=sel_rows["pressure_Pa"].to_numpy()
    )

    # elevation gate (final)
    elev_ok = (el >= ELEV_LIMITS_DEG[0]) & (el <= ELEV_LIMITS_DEG[1])
    obs_uv = obs_uv[elev_ok]
    times = times[elev_ok]
    w = w[elev_ok]
    az = az[elev_ok]
    el = el[elev_ok]
    if len(obs_uv) < 60:
        raise RuntimeError("Insufficient frames after elevation gating.")

    sun_world = unit_vector_from_az_el(az, el)

    # --- initial KB parameters ---
    f0 = 2*R0/np.pi  # equidistant-ish initial guess
    x0 = np.array([cx0, cy0, f0, 0.0, 0.0, 0.0, 0.0,  # intrinsics
                   0.0, 0.0, 0.0], dtype=np.float64)  # yaw, pitch, roll

    def residuals(x):
        cx, cy, f, k1,k2,k3,k4, yaw,pitch,roll = x
        R = rot_from_yaw_pitch_roll(yaw, pitch, roll)
        v_cam = (R @ sun_world.T).T
        proj = project_dir_to_px(v_cam, cx, cy, f, k1,k2,k3,k4)
        res = (proj - obs_uv).reshape(-1,2)
        # weight by DNI and normalize coordinate scale
        res = (res * w[:,None]).ravel()
        return res

    res = least_squares(residuals, x0, method="trf", verbose=2, max_nfev=2000)
    cx, cy, f, k1,k2,k3,k4, yaw,pitch,roll = res.x
    K = np.array([[f,0,cx],[0,f,cy],[0,0,1]], dtype=np.float64)
    D = np.array([k1,k2,k3,k4], dtype=np.float64)
    R_world2cam = rot_from_yaw_pitch_roll(yaw, pitch, roll)

    print("\n--- Calibration summary ---")
    print("K=\n", K)
    print("D=", D)
    print("R_world2cam=\n", R_world2cam)
    print("RMS pixel error (weighted): {:.2f}".format(math.sqrt(res.cost/len(obs_uv))))

    # Save like OpenCV tutorials do
    fs = cv.FileStorage("fisheye_calib.yml", cv.FILE_STORAGE_WRITE)
    fs.write("K", K); fs.write("D", D); fs.write("R_world2cam", R_world2cam)
    fs.write("image_size", np.array(shape[::-1], dtype=np.int32))
    fs.write("notes", "Calibrated from sun track using MESOR; K is fisheye-KB focal in pixels.")
    fs.release()

    return K, D, R_world2cam, shape

def undistort_example(sample_image_path, K, D, balance=0.2, dim_out=None):
    img = cv.imread(sample_image_path, cv.IMREAD_COLOR)
    h, w = img.shape[:2]
    if dim_out is None:
        dim_out = (w, h)
    newK = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w,h), np.eye(3), balance=balance, new_size=dim_out
    )
    map1, map2 = cv.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), newK, dim_out, cv.CV_16SC2
    )
    und = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    cv.imwrite("undistorted_example.jpg", und)
    print("Wrote undistorted_example.jpg")
    return und

if __name__ == "__main__":
    # df, meta = read_mesor(MESOR_PATH, LOCAL_TZ)
    K, D, R, shape = calibrate_with_mesor(
        IMG_GLOB, MESOR_PATH, LOCAL_TZ, SAMPLE_STRIDE, MAX_IMAGES
    )
    # pick a mid-day image for the demo
    # files = sorted(glob(IMG_GLOB))
    # if files:
    #     undistort_example(files[len(files)//2], K, D, balance=0.2)
