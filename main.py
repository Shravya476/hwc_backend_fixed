from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import math
import random
import time
import os
import requests
import hmac
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo

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

BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")

OTP_SECRET = os.environ.get("OTP_SECRET")
OTP_STEP_SECONDS = 300

IST = ZoneInfo("Asia/Kolkata")

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

# ── Named forest zones: kept for location naming + time-profile lookup ──
# (lat, lon, radius, NDVI, elev, slope, df, dw, dr, name, species_profile)
FOREST_ZONES = [
    (11.9, 76.1, 1.0,  0.82, 900,  12, 0.2, 0.4, 1.5, "Nagarhole", "elephant"),
    (11.6, 76.4, 1.0,  0.80, 860,  15, 0.2, 0.3, 1.8, "Bandipur", "elephant"),
    (11.6, 76.1, 1.0,  0.85, 790,  18, 0.1, 0.2, 1.6, "Wayanad", "elephant"),
    (12.4, 75.7, 1.0,  0.84, 1000, 22, 0.2, 0.3, 2.0, "Kodagu", "elephant"),
    (13.1, 75.3, 0.8,  0.86, 1100, 26, 0.2, 0.3, 2.5, "Kudremukh", "carnivore"),
    (11.9, 77.0, 0.8,  0.76, 1050, 20, 0.3, 0.5, 2.0, "BRT Hills", "elephant"),
    (11.5, 77.2, 0.8,  0.74, 820,  16, 0.3, 0.5, 2.2, "Sathyamangalam", "elephant"),
    (10.5, 76.9, 0.8,  0.80, 880,  17, 0.2, 0.4, 2.0, "Anamalai", "elephant"),
    (11.4, 76.7, 0.8,  0.78, 1100, 24, 0.2, 0.3, 2.5, "Nilgiris", "elephant"),
    (13.5, 75.7, 0.8,  0.77, 830,  14, 0.2, 0.4, 2.0, "Bhadra", "carnivore"),
    (13.3, 75.8, 0.7,  0.80, 900,  16, 0.2, 0.3, 2.0, "Chikmagalur Forest", "carnivore"),
    (12.6, 75.7, 0.7,  0.83, 1000, 22, 0.2, 0.3, 2.0, "Pushpagiri", "carnivore"),
    (14.0, 74.8, 0.7,  0.80, 680,  16, 0.2, 0.2, 2.5, "Sharavathi", "carnivore"),
    (13.4, 75.1, 0.7,  0.85, 820,  22, 0.2, 0.3, 2.5, "Agumbe", "carnivore"),
    (14.6, 74.8, 0.7,  0.78, 600,  18, 0.3, 0.4, 3.0, "Sirsi", "carnivore"),
    (11.2, 77.5, 0.7,  0.79, 940,  20, 0.2, 0.3, 2.5, "Kalakad", "elephant"),
    (12.0, 75.5, 0.8,  0.80, 850,  18, 0.2, 0.3, 2.0, "Coorg Buffer", "elephant"),
    (12.5, 76.0, 0.7,  0.81, 870,  19, 0.2, 0.3, 2.0, "Kabini", "elephant"),
    (12.2, 75.9, 0.7,  0.82, 880,  20, 0.2, 0.3, 2.0, "Brahmagiri", "elephant"),
    (11.0, 76.5, 0.7,  0.78, 820,  16, 0.2, 0.4, 2.0, "Palakkad Gap", "elephant"),
    (10.8, 76.7, 0.7,  0.76, 750,  14, 0.3, 0.4, 2.0, "Silent Valley", "carnivore"),
    (15.2, 74.6, 0.7,  0.79, 580,  17, 0.3, 0.4, 2.5, "Dandeli", "carnivore"),
    (12.4, 76.0, 0.9,  0.82, 860,  18, 0.2, 0.3, 2.0, "Namdroling Area", "elephant"),
]

# ── Coarse regional fallback profile when far from any named zone ──
# (min_lat, max_lat, min_lon, max_lon, profile)
REGION_PROFILES = [
    (8.0, 16.0, 74.0, 77.5, "elephant"),     # Western Ghats belt
    (18.0, 24.0, 77.0, 83.0, "carnivore"),   # Central India tiger landscape
    (26.0, 30.0, 77.0, 89.0, "elephant"),    # Terai / Himalayan foothills
    (21.0, 23.0, 88.0, 90.5, "diurnal_worker"),  # Sundarbans coastal
    (24.0, 29.0, 89.0, 97.0, "carnivore"),   # Northeast India
]

# ── Time-of-day multipliers by species behavior (hour 0-23) ──
TIME_PROFILES = {
    "elephant": [0.80,0.70,0.70,0.75,0.90,1.30,1.45,1.35,1.00,0.80,0.70,0.65,
                 0.60,0.65,0.70,0.80,0.95,1.20,1.45,1.40,1.20,1.00,0.90,0.85],
    "carnivore": [1.30,1.40,1.45,1.35,1.20,1.00,0.85,0.75,0.70,0.65,0.60,0.60,
                  0.60,0.60,0.65,0.70,0.80,0.90,1.00,1.10,1.20,1.25,1.35,1.40],
    "diurnal_worker": [0.60,0.55,0.55,0.60,0.70,0.90,1.10,1.30,1.40,1.35,1.25,1.15,
                       1.05,1.10,1.20,1.25,1.15,1.00,0.85,0.70,0.65,0.60,0.58,0.58],
    "mixed": [0.95,0.90,0.90,0.92,0.95,1.05,1.10,1.15,1.05,1.00,0.95,0.92,
              0.90,0.92,0.95,1.00,1.05,1.10,1.15,1.10,1.05,1.00,0.97,0.95],
}

# Cache OSM/elevation lookups briefly to reduce repeat calls for popular spots
_terrain_cache: dict = {}
_CACHE_TTL = 24 * 3600

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def is_urban(lat, lon):
    for ulat, ulon, radius, name in URBAN_ZONES:
        if haversine_km(lat, lon, ulat, ulon) <= radius * 111:  # rough deg->km
            return True, name
    return False, None

def get_real_terrain(lat, lon):
    """Query OpenStreetMap (Overpass) + Open-Elevation for real nearby
    forest/water/road distances and real elevation. Raises on failure so
    the caller can fall back to the estimate-based method."""
    key = (round(lat, 3), round(lon, 3))
    cached = _terrain_cache.get(key)
    if cached and time.time() - cached["t"] < _CACHE_TTL:
        return cached["data"]

    query = f"""
    [out:json][timeout:15];
    (
      way["natural"="wood"](around:8000,{lat},{lon});
      way["landuse"="forest"](around:8000,{lat},{lon});
      way["natural"="water"](around:8000,{lat},{lon});
      way["waterway"](around:8000,{lat},{lon});
      way["highway"](around:5000,{lat},{lon});
    );
    out center;
    """
    resp = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        timeout=20,
    )
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    dist_forest, dist_water, dist_road = None, None, None
    for el in elements:
        center = el.get("center")
        if not center:
            continue
        d = haversine_km(lat, lon, center["lat"], center["lon"])
        tags = el.get("tags", {})
        if tags.get("natural") == "wood" or tags.get("landuse") == "forest":
            dist_forest = d if dist_forest is None else min(dist_forest, d)
        elif tags.get("natural") == "water" or "waterway" in tags:
            dist_water = d if dist_water is None else min(dist_water, d)
        elif "highway" in tags:
            dist_road = d if dist_road is None else min(dist_road, d)

    dist_forest = 9.0 if dist_forest is None else dist_forest
    dist_water  = 5.0 if dist_water  is None else dist_water
    dist_road   = 0.4 if dist_road   is None else dist_road

    elev_resp = requests.post(
        "https://api.open-elevation.com/api/v1/lookup",
        json={"locations": [
            {"latitude": lat, "longitude": lon},
            {"latitude": lat + 0.003, "longitude": lon},
            {"latitude": lat, "longitude": lon + 0.003},
        ]},
        timeout=15,
    )
    elev_resp.raise_for_status()
    results = elev_resp.json()["results"]
    elevation = results[0]["elevation"]
    dz_lat = abs(results[1]["elevation"] - elevation)
    dz_lon = abs(results[2]["elevation"] - elevation)
    slope = min(45.0, (max(dz_lat, dz_lon) / 300.0) * 100)  # rough % grade

    NDVI = max(0.15, min(0.88, 0.85 - dist_forest * 0.08))
    NDWI = max(0.05, min(0.35, 0.35 - dist_water * 0.05))

    data = (NDVI, NDWI, elevation, slope, dist_forest, dist_water, dist_road)
    _terrain_cache[key] = {"t": time.time(), "data": data}
    return data

def estimate_features_fallback(lat, lon):
    """Old zone-blend estimate, used only if live OSM/elevation calls fail."""
    in_ghats = (74.0 <= lon <= 77.5) and (10.0 <= lat <= 15.5)
    best_dist, best_zone = float('inf'), None
    for zone in FOREST_ZONES:
        d = math.sqrt((lat - zone[0])**2 + (lon - zone[1])**2)
        if d < best_dist:
            best_dist, best_zone = d, zone

    if best_dist <= best_zone[2]:
        blend = 1.0 - (best_dist / best_zone[2])
        NDVI  = best_zone[3]*blend + 0.18*(1-blend)
        NDWI  = 0.35*blend + 0.08*(1-blend)
        elev  = best_zone[4]*blend + 300*(1-blend)
        slope = best_zone[5]*blend + 2*(1-blend)
        df    = best_zone[6]*blend + 9.0*(1-blend)
        dw    = best_zone[7]*blend + 5.0*(1-blend)
        dr    = best_zone[8]*blend + 0.4*(1-blend)
    elif in_ghats:
        f = min(1.0, best_dist / 2.0)
        NDVI, NDWI, elev, slope = 0.60-f*0.25, 0.30-f*0.10, 600-f*300, 12-f*8
        df, dw, dr = 1.5+f*3.0, 0.8+f*2.0, 2.0-f*1.0
    else:
        NDVI, NDWI, elev, slope, df, dw, dr = 0.18, 0.08, 400, 2, 9.0, 5.0, 0.4
    return NDVI, NDWI, elev, slope, df, dw, dr

def build_features(lat, lon, NDVI, NDWI, elevation, slope, dist_forest, dist_water, dist_road):
    VWR = NDVI / (NDWI + 0.01)
    TRI = slope * 0.5
    NLD = min(1, (lon - 75) / 5)
    HAS = min(1, (lat - 10) / 5)
    ESI = (NDVI + NDWI) / 2
    NDVI_NDWI_interaction = NDVI * NDWI
    veg_water_risk = NDVI / (dist_water + 0.1)
    isolation_index = (dist_forest + dist_road) / 2
    terrain_ratio = slope / (elevation + 1)
    human_pressure = (HAS + NLD) / 2
    eco_stress = (NDVI + NDWI + HAS) / 3
    slope_elev_risk = (slope * elevation) / 1000
    return [lat, lon, NDVI, NDWI, elevation, slope, dist_forest, dist_water,
            dist_road, VWR, TRI, NLD, HAS, ESI, NDVI_NDWI_interaction,
            veg_water_risk, isolation_index, terrain_ratio, human_pressure,
            eco_stress, slope_elev_risk]

def get_location_name(lat, lon):
    places = [(z[0], z[1], z[9]) for z in FOREST_ZONES]
    places += [(u[0], u[1], u[3]) for u in URBAN_ZONES]
    best_dist, best_name = float('inf'), f"({lat:.2f}, {lon:.2f})"
    for plat, plon, name in places:
        d = math.sqrt((lat - plat)**2 + (lon - plon)**2)
        if d < best_dist:
            best_dist, best_name = d, name
    if best_dist > 0.6:
        best_name = f"({lat:.2f}, {lon:.2f})"
    return best_name

def get_time_profile(lat, lon):
    best_dist, best_profile = float('inf'), None
    for zone in FOREST_ZONES:
        d = math.sqrt((lat - zone[0])**2 + (lon - zone[1])**2)
        if d < best_dist:
            best_dist, best_profile = d, zone[10]
    if best_dist <= 0.5:
        return best_profile
    for min_lat, max_lat, min_lon, max_lon, profile in REGION_PROFILES:
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return profile
    return "mixed"

def run_prediction(lat: float, lon: float, hour: Optional[int] = None):
    urban, urban_name = is_urban(lat, lon)
    if urban:
        return {
            "risk": "LOW", "probability": 5.0, "location": urban_name,
            "driver": "human_pressure", "time_profile": "urban",
            "hour_used": hour if hour is not None else datetime.now(IST).hour,
            "lat": lat, "lon": lon,
        }

    used_fallback = False
    try:
        NDVI, NDWI, elev, slope, df, dw, dr = get_real_terrain(lat, lon)
    except Exception:
        NDVI, NDWI, elev, slope, df, dw, dr = estimate_features_fallback(lat, lon)
        used_fallback = True

    feats = build_features(lat, lon, NDVI, NDWI, elev, slope, df, dw, dr)
    dfx = pd.DataFrame([feats], columns=FEATURES)
    base_prob = float(model.predict_proba(scaler.transform(dfx))[0][1] * 100)

    hour_used = hour if hour is not None else datetime.now(IST).hour
    hour_used = max(0, min(23, hour_used))
    profile = get_time_profile(lat, lon)
    multiplier = TIME_PROFILES[profile][hour_used]
    adjusted_prob = max(0.0, min(100.0, base_prob * multiplier))

    risk = "HIGH" if adjusted_prob >= 70 else "MEDIUM" if adjusted_prob >= 40 else "LOW"
    location = get_location_name(lat, lon)
    driver = "NDVI" if NDVI > 0.65 else "dist_forest"

    return {
        "risk": risk,
        "probability": round(adjusted_prob, 2),
        "base_probability": round(base_prob, 2),
        "time_multiplier": round(multiplier, 2),
        "time_profile": profile,
        "hour_used": hour_used,
        "location": location,
        "driver": driver,
        "used_fallback_terrain": used_fallback,
        "lat": lat,
        "lon": lon,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"name": "HWC Prediction API", "status": "live"}

@app.get("/predict")
def predict_get(lat: float, lon: float, hour: Optional[int] = None):
    return run_prediction(lat, lon, hour)

class PredictRequest(BaseModel):
    lat: float
    lon: float
    hour: Optional[int] = None

@app.post("/predict")
def predict_post(req: PredictRequest):
    return run_prediction(req.lat, req.lon, req.hour)

class EmailRequest(BaseModel):
    email: str

class VerifyRequest(BaseModel):
    email: str
    otp: str

def send_email_otp(to_email: str, otp: str):
    response = requests.post(
        "https://api.brevo.com/v3/smtp/email",
        headers={
            "api-key": BREVO_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "sender": {"email": SENDER_EMAIL, "name": "HWC Alert"},
            "to": [{"email": to_email}],
            "subject": "HWC Alert - Your Login Code",
            "textContent": f"Your HWC Alert verification code is: {otp}\n\nExpires in 5 minutes.",
        },
        timeout=15,
    )
    if response.status_code >= 300:
        raise Exception(f"Brevo API error {response.status_code}: {response.text}")

def generate_otp(email: str, time_step: int) -> str:
    msg = f"{email.lower().strip()}:{time_step}".encode()
    digest = hmac.new(OTP_SECRET.encode(), msg, hashlib.sha256).hexdigest()
    return f"{int(digest, 16) % 1000000:06d}"

@app.post("/send-otp")
def send_otp(req: EmailRequest):
    time_step = int(time.time() // OTP_STEP_SECONDS)
    otp = generate_otp(req.email, time_step)
    try:
        send_email_otp(req.email, otp)
    except Exception as e:
        return {"error": f"Failed to send email: {e}"}
    return {"message": "OTP sent"}

@app.post("/verify-otp")
def verify_otp(req: VerifyRequest):
    current_step = int(time.time() // OTP_STEP_SECONDS)
    entered = req.otp.strip()
    if generate_otp(req.email, current_step) == entered or generate_otp(req.email, current_step - 1) == entered:
        return {"message": "Verified"}
    return {"error": "Incorrect or expired OTP"}
