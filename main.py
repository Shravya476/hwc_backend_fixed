from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import math
import random
import time
import smtplib
from email.mime.text import MIMEText

app = FastAPI(title="HWC Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model    = joblib.load("P5model.pkl")
scaler   = joblib.load("P5scaler.pkl")
FEATURES = joblib.load("P5feature_columns.pkl")

otp_store = {}  # email -> (otp, expiry_timestamp)

SMTP_EMAIL = "jannu3385@gmail.com"
SMTP_PASSWORD = "SravyaHindu@_0508"

# ── Urban zones → always LOW ─────────────────────────────────────
URBAN_ZONES = [
    (12.90, 77.50, 0.6, "Bengaluru"),
    (12.97, 77.59, 0.6, "Bengaluru"),
    (12.30, 76.64, 0.4, "Mysuru"),
    (12.87, 74.88, 0.3, "Mangaluru"),
    (13.00, 76.10, 0.3, "Hassan"),
    (11.02, 76.96, 0.3, "Coimbatore"),
    (10.52, 76.21, 0.3, "Thrissur"),
    (13.34, 77.10, 0.3, "Tumkur"),
    (12.52, 76.90, 0.3, "Mandya"),
    (11.34, 77.72, 0.3, "Erode"),
    (11.65, 78.15, 0.3, "Salem"),
    (10.00, 77.00, 0.3, "Madurai"),
    (13.93, 75.57, 0.3, "Shimoga City"),
    (12.72, 77.28, 0.3, "Ramanagara"),
    (12.65, 77.20, 0.3, "Channapatna"),
    (13.13, 78.13, 0.3, "Kolar"),
    (13.43, 77.73, 0.3, "Chikkaballapur"),
    (15.14, 76.92, 0.3, "Bellary"),
    (14.47, 75.92, 0.3, "Davangere"),
    (15.13, 75.71, 0.3, "Hubli"),
    (10.80, 76.65, 0.3, "Palakkad City"),
    (11.25, 75.77, 0.3, "Kozhikode"),
    (10.00, 76.96, 0.3, "Kochi"),
]

# ── Forest zones → HIGH risk (expanded radii) ────────────────────
FOREST_ZONES = [
    # lat,  lon,  radius, NDVI, elev, slope, df,  dw,  dr,  name
    (11.9, 76.1, 1.0,  0.82, 900,  12, 0.2, 0.4, 1.5, "Nagarhole"),
    (11.6, 76.4, 1.0,  0.80, 860,  15, 0.2, 0.3, 1.8, "Bandipur"),
    (11.6, 76.1, 1.0,  0.85, 790,  18, 0.1, 0.2, 1.6, "Wayanad"),
    (12.4, 75.7, 1.0,  0.84, 1000, 22, 0.2, 0.3, 2.0, "Kodagu"),
    (13.1, 75.3, 0.8,  0.86, 1100, 26, 0.2, 0.3, 2.5, "Kudremukh"),
    (11.9, 77.0, 0.8,  0.76, 1050, 20, 0.3, 0.5, 2.0, "BRT Hills"),
    (11.5, 77.2, 0.8,  0.74, 820,  16, 0.3, 0.5, 2.2, "Sathyamangalam"),
    (10.5, 76.9, 0.8,  0.80, 880,  17, 0.2, 0.4, 2.0, "Anamalai"),
    (11.4, 76.7, 0.8,  0.78, 1100, 24, 0.2, 0.3, 2.5, "Nilgiris"),
    (13.5, 75.7, 0.8,  0.77, 830,  14, 0.2, 0.4, 2.0, "Bhadra"),
    (13.3, 75.8, 0.7,  0.80, 900,  16, 0.2, 0.3, 2.0, "Chikmagalur Forest"),
    (12.6, 75.7, 0.7,  0.83, 1000, 22, 0.2, 0.3, 2.0, "Pushpagiri"),
    (14.0, 74.8, 0.7,  0.80, 680,  16, 0.2, 0.2, 2.5, "Sharavathi"),
    (13.4, 75.1, 0.7,  0.85, 820,  22, 0.2, 0.3, 2.5, "Agumbe"),
    (14.6, 74.8, 0.7,  0.78, 600,  18, 0.3, 0.4, 3.0, "Sirsi"),
    (11.2, 77.5, 0.7,  0.79, 940,  20, 0.2, 0.3, 2.5, "Kalakad"),
    (12.0, 75.5, 0.8,  0.80, 850,  18, 0.2, 0.3, 2.0, "Coorg Buffer"),
    (12.5, 76.0, 0.7,  0.81, 870,  19, 0.2, 0.3, 2.0, "Kabini"),
    (12.2, 75.9, 0.7,  0.82, 880,  20, 0.2, 0.3, 2.0, "Brahmagiri"),
    (11.0, 76.5, 0.7,  0.78, 820,  16, 0.2, 0.4, 2.0, "Palakkad Gap"),
    (10.8, 76.7, 0.7,  0.76, 750,  14, 0.3, 0.4, 2.0, "Silent Valley"),
    (15.2, 74.6, 0.7,  0.79, 580,  17, 0.3, 0.4, 2.5, "Dandeli"),
    (12.4, 76.0, 0.9,  0.82, 860,  18, 0.2, 0.3, 2.0, "Namdroling Area"),
]

def is_urban(lat, lon):
    for ulat, ulon, radius, name in URBAN_ZONES:
        dist = math.sqrt((lat - ulat)**2 + (lon - ulon)**2)
        if dist <= radius:
            return True, name
    return False, None

def estimate_features(lat, lon):
    in_ghats = (74.0 <= lon <= 77.5) and (10.0 <= lat <= 15.5)

    best_dist = float('inf')
    best_zone = None
    for zone in FOREST_ZONES:
        dist = math.sqrt((lat - zone[0])**2 + (lon - zone[1])**2)
        if dist < best_dist:
            best_dist = dist
            best_zone = zone

    if best_dist <= best_zone[2]:
        blend = 1.0 - (best_dist / best_zone[2])
        NDVI  = best_zone[3] * blend + 0.18 * (1 - blend)
        NDWI  = 0.35 * blend + 0.08 * (1 - blend)
        elev  = best_zone[4] * blend + 300  * (1 - blend)
        slope = best_zone[5] * blend + 2    * (1 - blend)
        df    = best_zone[6] * blend + 9.0  * (1 - blend)
        dw    = best_zone[7] * blend + 5.0  * (1 - blend)
        dr    = best_zone[8] * blend + 0.4  * (1 - blend)
    elif in_ghats:
        dist_factor = min(1.0, best_dist / 2.0)
        NDVI  = 0.60 - dist_factor * 0.25
        NDWI  = 0.30 - dist_factor * 0.10
        elev  = 600  - dist_factor * 300
        slope = 12   - dist_factor * 8
        df    = 1.5  + dist_factor * 3.0
        dw    = 0.8  + dist_factor * 2.0
        dr    = 2.0  - dist_factor * 1.0
    else:
        NDVI  = 0.18; NDWI = 0.08; elev = 400; slope = 2
        df    = 9.0;  dw   = 5.0;  dr   = 0.4

    return NDVI, NDWI, elev, slope, df, dw, dr

def build_features(lat, lon, NDVI, NDWI, elevation, slope,
                   dist_forest, dist_water, dist_road):
    VWR                   = NDVI / (NDWI + 0.01)
    TRI                   = slope * 0.5
    NLD                   = min(1, (lon - 75) / 5)
    HAS                   = min(1, (lat - 10) / 5)
    ESI                   = (NDVI + NDWI) / 2
    NDVI_NDWI_interaction = NDVI * NDWI
    veg_water_risk        = NDVI / (dist_water + 0.1)
    isolation_index       = (dist_forest + dist_road) / 2
    terrain_ratio         = slope / (elevation + 1)
    human_pressure        = (HAS + NLD) / 2
    eco_stress            = (NDVI + NDWI + HAS) / 3
    slope_elev_risk       = (slope * elevation) / 1000
    return [lat, lon, NDVI, NDWI, elevation, slope, dist_forest, dist_water,
            dist_road, VWR, TRI, NLD, HAS, ESI, NDVI_NDWI_interaction,
            veg_water_risk, isolation_index, terrain_ratio, human_pressure,
            eco_stress, slope_elev_risk]

def get_location_name(lat, lon):
    places = [(z[0], z[1], z[9]) for z in FOREST_ZONES]
    places += [(u[0], u[1], u[3]) for u in URBAN_ZONES]
    best_dist = float('inf')
    best_name = f"({lat:.2f}, {lon:.2f})"
    for plat, plon, name in places:
        dist = math.sqrt((lat - plat)**2 + (lon - plon)**2)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if best_dist > 0.6:
        best_name = f"({lat:.2f}, {lon:.2f})"
    return best_name

def run_prediction(lat: float, lon: float):
    urban, urban_name = is_urban(lat, lon)
    if urban:
        return {
            "risk":        "LOW",
            "probability": 5.0,
            "location":    urban_name,
            "driver":      "human_pressure",
            "lat":         lat,
            "lon":         lon,
        }

    NDVI, NDWI, elev, slope, df, dw, dr = estimate_features(lat, lon)
    feats    = build_features(lat, lon, NDVI, NDWI, elev, slope, df, dw, dr)
    dfx      = pd.DataFrame([feats], columns=FEATURES)
    prob     = float(model.predict_proba(scaler.transform(dfx))[0][1] * 100)
    risk     = "HIGH" if prob >= 70 else "MEDIUM" if prob >= 40 else "LOW"
    location = get_location_name(lat, lon)
    driver   = "NDVI" if NDVI > 0.65 else "dist_forest"

    return {
        "risk":        risk,
        "probability": round(prob, 2),
        "location":    location,
        "driver":      driver,
        "lat":         lat,
        "lon":         lon,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"name": "HWC Prediction API", "status": "live"}

@app.get("/predict")
def predict_get(lat: float, lon: float):
    return run_prediction(lat, lon)

class PredictRequest(BaseModel):
    lat: float
    lon: float

@app.post("/predict")
def predict_post(req: PredictRequest):
    return run_prediction(req.lat, req.lon)

class EmailRequest(BaseModel):
    email: str

class VerifyRequest(BaseModel):
    email: str
    otp: str

def send_email_otp(to_email: str, otp: str):
    msg = MIMEText(f"Your HWC Alert verification code is: {otp}\n\nExpires in 5 minutes.")
    msg["Subject"] = "HWC Alert - Your Login Code"
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, to_email, msg.as_string())

@app.post("/send-otp")
def send_otp(req: EmailRequest):
    otp = str(random.randint(100000, 999999))
    otp_store[req.email] = (otp, time.time() + 300)
    try:
        send_email_otp(req.email, otp)
    except Exception as e:
        return {"error": f"Failed to send email: {e}"}
    return {"message": "OTP sent"}

@app.post("/verify-otp")
def verify_otp(req: VerifyRequest):
    record = otp_store.get(req.email)
    if not record:
        return {"error": "No OTP requested for this email"}
    otp, expiry = record
    if time.time() > expiry:
        del otp_store[req.email]
        return {"error": "OTP expired"}
    if req.otp != otp:
        return {"error": "Incorrect OTP"}
    del otp_store[req.email]
    return {"message": "Verified"}
