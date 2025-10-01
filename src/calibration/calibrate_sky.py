import re, glob, numpy as np, cv2 as cv, pandas as pd
from scipy.optimize import least_squares
# from pvlib.solarposition import get_solarposition
import pvlib
from datetime import timezone

# ---- Site info (set to your numbers) ----
LAT, LON, ALT = 37.0916, -2.3636, 490.587   # PSA Metas approx
TZ = 'Europe/Madrid'                    # check your timestamp basis (local vs UTC)

# ---- Utilities ----
def parse_ts_from_name(fn):         # 20191001130000
    m = re.search(r'(\d{14})', fn)  # e.g. 20190908120100
    ts = pd.to_datetime(m.group(1), format='%Y%m%d%H%M%S', utc=True)
    # If filenames are local clock, remove 'utc=True' and localize to TZ instead.
    return ts.tz_convert(TZ)

def find_sun_xy(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thr  = np.percentile(gray, 99.5)
    mask = (gray >= thr).astype(np.uint8)*255
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    cnts,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv.contourArea)
    (x,y),r = cv.minEnclosingCircle(c)
    return np.array([x,y], float)

def sph_to_dir(zenith, azimuth):
    # Local frame: z up, x east, y north (pvlib azimuth: 0=N, 90=E)
    az = np.deg2rad(azimuth); th = np.deg2rad(zenith)
    x = np.sin(th)*np.sin(az)
    y = np.sin(th)*np.cos(az)
    z = np.cos(th)
    return np.stack([x,y,z], -1)

def euler_to_R(yaw, pitch, roll):
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx

def project_fisheye(K, D, R, dirs):
    # dirs: Nx3 unit vectors in world frame
    pts = (R @ dirs.T).T              # rotate into camera frame
    # normalize (x/z, y/z) for a unit sphere projection onto image plane
    x = pts[:,0] / (pts[:,2] + 1e-12)
    y = pts[:,1] / (pts[:,2] + 1e-12)
    r2 = x*x + y*y
    k1,k2,k3,k4 = D
    scale = 1 + k1*r2 + k2*r2**2 + k3*r2**3 + k4*r2**4
    xd, yd = x*scale, y*scale
    fx, fy, cx, cy = K
    u = fx*xd + cx
    v = fy*yd + cy
    return np.stack([u,v], -1)

# ---- Collect (u,v) and sun directions ----
rows = []
for fn in sorted(glob.glob('../../data/images/raw/*.jpg')):   # point to your folder
    ts = parse_ts_from_name(fn)
    img = cv.imread(fn)
    uv  = find_sun_xy(img)
    print(fn, ' - ', uv)
    if uv is None: continue
    sp = pvlib.solarposition.get_solarposition(ts, LAT, LON, ALT)
    zen, az = float(sp['zenith']), float(sp['azimuth'])
    d = sph_to_dir(zen, az)
    rows.append((uv, d, img.shape[1], img.shape[0]))

uv = np.array([r[0] for r in rows])
dirs = np.array([r[1] for r in rows])
W, H = rows[0][2], rows[0][3]

# ---- Initial guesses ----
cx0, cy0 = W/2, H/2
fx0 = fy0 = 500.0
p0 = np.array([fx0, fy0, cx0, cy0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [fx,fy,cx,cy,k1..k4,yaw,pitch,roll]

def residuals(p):
    fx,fy,cx,cy,k1,k2,k3,k4,yaw,pitch,roll = p
    K = (fx,fy,cx,cy); D = (k1,k2,k3,k4); R = euler_to_R(yaw,pitch,roll)
    uv_hat = project_fisheye(K, D, R, dirs)
    return (uv_hat - uv).ravel()

sol = least_squares(residuals, p0, verbose=2, max_nfev=200)
fx,fy,cx,cy,k1,k2,k3,k4,yaw,pitch,roll = sol.x
print("K:", np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]))
print("D:", [k1,k2,k3,k4])
print("R (yaw,pitch,roll):", [yaw,pitch,roll])
