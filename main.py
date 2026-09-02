from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

app = FastAPI(title="HWC Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("P5model.pkl")
scaler = joblib.load("P5scaler.pkl")
FEATURES = joblib.load("P5feature_columns.pkl")

BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")  # must match your verified Brevo sender
OTP_SECRET = os.environ.get("OTP_SECRET")      # any long random string, set on Render
OTP_STEP_SECONDS = 300  # each code is valid for a 5-minute window

# ── Urban zones → always LOW ─────────────────────────────────────
URBAN_ZONES = [
    # Karnataka / Kerala / Tamil Nadu (original coverage)
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
    (8.52, 76.94, 0.3, "Thiruvananthapuram"),
    (9.93, 76.27, 0.3, "Kottayam"),
    (13.08, 80.27, 0.5, "Chennai"),
    (12.82, 80.04, 0.3, "Chengalpattu"),
    (11.66, 78.15, 0.3, "Namakkal"),
    (9.93, 78.12, 0.3, "Madurai South"),
    (8.09, 77.55, 0.3, "Nagercoil"),

    # North India
    (28.61, 77.21, 0.6, "Delhi"),
    (28.46, 77.03, 0.4, "Gurugram"),
    (28.57, 77.32, 0.4, "Noida"),
    (26.91, 75.79, 0.4, "Jaipur"),
    (26.45, 80.33, 0.4, "Kanpur"),
    (26.85, 80.95, 0.4, "Lucknow"),
    (25.32, 82.99, 0.4, "Varanasi"),
    (25.59, 85.14, 0.4, "Patna"),
    (30.32, 78.03, 0.3, "Dehradun"),
    (31.10, 77.17, 0.3, "Shimla"),
    (30.73, 76.78, 0.3, "Chandigarh"),
    (34.08, 74.79, 0.3, "Srinagar"),
    (32.73, 74.86, 0.3, "Jammu"),
    (27.18, 78.02, 0.3, "Agra"),
    (26.23, 78.18, 0.3, "Gwalior"),
    (23.18, 79.99, 0.3, "Jabalpur"),
    (23.26, 77.41, 0.3, "Bhopal"),
    (22.72, 75.86, 0.3, "Indore"),
    (21.15, 79.09, 0.4, "Nagpur"),

    # West India
    (19.08, 72.88, 0.6, "Mumbai"),
    (18.52, 73.86, 0.4, "Pune"),
    (21.17, 72.83, 0.4, "Surat"),
    (23.02, 72.57, 0.4, "Ahmedabad"),
    (22.31, 70.80, 0.3, "Rajkot"),
    (23.22, 72.68, 0.3, "Gandhinagar"),
    (25.20, 75.86, 0.3, "Kota"),
    (24.58, 73.68, 0.3, "Udaipur"),
    (26.28, 73.02, 0.3, "Jodhpur"),

    # East / Northeast India
    (22.57, 88.36, 0.5, "Kolkata"),
    (20.30, 85.82, 0.4, "Bhubaneswar"),
    (23.34, 85.31, 0.3, "Ranchi"),
    (21.25, 81.63, 0.3, "Raipur"),
    (26.14, 91.74, 0.4, "Guwahati"),
    (25.58, 91.89, 0.3, "Shillong"),
    (23.83, 91.28, 0.3, "Agartala"),
    (24.66, 93.90, 0.3, "Imphal"),
    (25.67, 94.11, 0.3, "Kohima"),
    (23.71, 92.72, 0.3, "Aizawl"),
    (27.48, 94.90, 0.3, "Dibrugarh"),

    # South / Andhra / Telangana
    (17.38, 78.49, 0.5, "Hyderabad"),
    (17.69, 83.22, 0.4, "Visakhapatnam"),
    (16.51, 80.62, 0.3, "Vijayawada"),
    (15.83, 78.04, 0.3, "Kurnool"),
    (14.44, 79.99, 0.3, "Nellore"),
]

# ── Forest zones → HIGH risk (expanded radii) ────────────────────
FOREST_ZONES = [
    # lat, lon, radius, NDVI, elev, slope, df, dw, dr, name
    (11.9, 76.1, 1.0, 0.82, 900, 12, 0.2, 0.4, 1.5, "Nagarhole"),
    (11.6, 76.4, 1.0, 0.80, 860, 15, 0.2, 0.3, 1.8, "Bandipur"),
    (11.6, 76.1, 1.0, 0.85, 790, 18, 0.1, 0.2, 1.6, "Wayanad"),
    (12.4, 75.7, 1.0, 0.84, 1000, 22, 0.2, 0.3, 2.0, "Kodagu"),
    (13.1, 75.3, 0.8, 0.86, 1100, 26, 0.2, 0.3, 2.5, "Kudremukh"),
    (11.9, 77.0, 0.8, 0.76, 1050, 20, 0.3, 0.5, 2.0, "BRT Hills"),
    (11.5, 77.2, 0.8, 0.74, 820, 16, 0.3, 0.5, 2.2, "Sathyamangalam"),
    (10.5, 76.9, 0.8, 0.80, 880, 17, 0.2, 0.4, 2.0, "Anamalai"),
    (11.4, 76.7, 0.8, 0.78, 1100, 24, 0.2, 0.3, 2.5, "Nilgiris"),
    (13.5, 75.7, 0.8, 0.77, 830, 14, 0.2, 0.4, 2.0, "Bhadra"),
    (13.3, 75.8, 0.7, 0.80, 900, 16, 0.2, 0.3, 2.0, "Chikmagalur Forest"),
    (12.6, 75.7, 0.7, 0.83, 1000, 22, 0.2, 0.3, 2.0, "Pushpagiri"),
    (14.0, 74.8, 0.7, 0.80, 680, 16, 0.2, 0.2, 2.5, "Sharavathi"),
    (13.4, 75.1, 0.7, 0.85, 820, 22, 0.2, 0.3, 2.5, "Agumbe"),
    (14.6, 74.8, 0.7, 0.78, 600, 18, 0.3, 0.4, 3.0, "Sirsi"),
    (11.2, 77.5, 0.7, 0.79, 940, 20, 0.2, 0.3, 2.5, "Kalakad"),
    (12.0, 75.5, 0.8, 0.80, 850, 18, 0.2, 0.3, 2.0, "Coorg Buffer"),
    (12.5, 76.0, 0.7, 0.81, 870, 19, 0.2, 0.3, 2.0, "Kabini"),
    (12.2, 75.9, 0.7, 0.82, 880, 20, 0.2, 0.3, 2.0, "Brahmagiri"),
    (11.0, 76.5, 0.7, 0.78, 820, 16, 0.2, 0.4, 2.0, "Palakkad Gap"),
    (10.8, 76.7, 0.7, 0.76, 750, 14, 0.3, 0.4, 2.0, "Silent Valley"),
    (15.2, 74.6, 0.7, 0.79, 580, 17, 0.3, 0.4, 2.5, "Dandeli"),
    (12.4, 76.0, 0.9, 0.82, 860, 18, 0.2, 0.3, 2.0, "Namdroling Area"),

    # ── Central India ────────────────────────────────────────────
    (22.33, 80.65, 1.0, 0.78, 600, 10, 0.2, 0.4, 2.0, "Kanha"),
    (23.65, 80.95, 1.0, 0.76, 550, 9, 0.2, 0.4, 2.0, "Bandhavgarh"),
    (21.75, 79.30, 1.0, 0.75, 500, 8, 0.2, 0.4, 2.0, "Pench"),
    (20.23, 79.35, 1.0, 0.74, 480, 8, 0.2, 0.4, 2.0, "Tadoba"),
    (22.50, 78.35, 1.1, 0.77, 650, 12, 0.2, 0.4, 2.0, "Satpura"),
    (21.30, 77.30, 1.0, 0.76, 700, 14, 0.2, 0.4, 2.0, "Melghat"),
    (22.50, 81.75, 0.9, 0.75, 600, 10, 0.2, 0.5, 2.5, "Achanakmar"),
    (19.30, 81.20, 0.9, 0.73, 450, 8, 0.3, 0.5, 3.0, "Indravati"),
    (18.85, 78.90, 0.9, 0.72, 400, 7, 0.3, 0.5, 3.0, "Kawal"),
    (20.90, 78.85, 0.8, 0.70, 420, 8, 0.3, 0.5, 2.5, "Navegaon-Nagzira"),
    (23.90, 81.95, 0.8, 0.72, 500, 9, 0.3, 0.5, 2.5, "Sanjay-Dubri"),

    # ── Himalayan / Terai belt ──────────────────────────────────
    (29.53, 78.77, 1.0, 0.75, 500, 10, 0.2, 0.3, 2.0, "Corbett"),
    (30.05, 78.20, 0.9, 0.72, 550, 12, 0.2, 0.3, 2.0, "Rajaji"),
    (28.52, 80.60, 0.9, 0.73, 200, 5, 0.2, 0.3, 2.0, "Dudhwa"),
    (27.35, 84.10, 0.8, 0.70, 180, 5, 0.3, 0.4, 2.5, "Valmiki"),
    (31.75, 77.60, 1.0, 0.68, 2500, 35, 0.3, 0.4, 4.0, "Great Himalayan NP"),
    (30.50, 79.10, 0.9, 0.62, 3000, 38, 0.4, 0.4, 5.0, "Kedarnath WLS"),
    (32.20, 77.15, 0.9, 0.60, 2800, 34, 0.4, 0.4, 4.5, "Pin Valley/Manali"),
    (29.35, 79.55, 0.9, 0.70, 1800, 25, 0.3, 0.4, 3.0, "Nanda Devi"),
    (30.85, 77.85, 0.8, 0.68, 2200, 30, 0.3, 0.4, 3.5, "Govind WLS"),

    # ── Northeast India ─────────────────────────────────────────
    (26.58, 93.30, 1.0, 0.80, 90, 4, 0.1, 0.2, 2.0, "Kaziranga"),
    (26.70, 91.00, 0.9, 0.78, 100, 5, 0.2, 0.3, 2.5, "Manas"),
    (26.97, 92.90, 0.8, 0.76, 120, 6, 0.2, 0.3, 2.5, "Nameri"),
    (27.50, 96.35, 0.9, 0.82, 300, 15, 0.2, 0.3, 3.5, "Namdapha"),
    (23.75, 92.35, 0.8, 0.77, 400, 18, 0.3, 0.4, 3.0, "Dampa"),
    (25.55, 93.35, 0.8, 0.75, 500, 20, 0.3, 0.4, 3.5, "Intanki"),
    (25.10, 93.75, 0.8, 0.74, 450, 18, 0.3, 0.4, 3.5, "Khangchendzonga Buffer"),
    (27.35, 88.60, 0.8, 0.72, 2000, 28, 0.3, 0.4, 3.5, "Neora Valley"),

    # ── Eastern India ────────────────────────────────────────────
    (21.60, 86.30, 0.9, 0.76, 400, 12, 0.2, 0.4, 2.5, "Simlipal"),
    (21.90, 88.90, 0.9, 0.65, 5, 1, 0.1, 0.1, 3.0, "Sundarbans"),
    (16.35, 78.85, 1.0, 0.72, 350, 10, 0.2, 0.4, 2.5, "Nagarjunsagar-Srisailam"),
    (20.70, 86.90, 0.8, 0.68, 10, 2, 0.2, 0.2, 2.5, "Bhitarkanika"),
    (22.10, 86.20, 0.7, 0.70, 300, 8, 0.3, 0.4, 3.0, "Dalma"),

    # ── Western India / Rajasthan / Gujarat ─────────────────────
    (26.02, 76.50, 0.9, 0.68, 350, 12, 0.3, 0.5, 2.0, "Ranthambore"),
    (27.30, 76.42, 0.8, 0.65, 400, 14, 0.3, 0.5, 2.5, "Sariska"),
    (21.15, 70.80, 0.9, 0.63, 200, 8, 0.3, 0.5, 2.5, "Gir"),
    (24.10, 79.85, 0.7, 0.62, 300, 10, 0.3, 0.5, 2.5, "Panna"),
    (26.90, 70.90, 1.0, 0.20, 250, 3, 0.6, 1.5, 3.0, "Desert NP"),
]


def is_urban(lat, lon):
    for ulat, ulon, radius, name in URBAN_ZONES:
        dist = math.sqrt((lat - ulat) ** 2 + (lon - ulon) ** 2)
        if dist <= radius:
            return True, name
    return False, None


def estimate_features(lat, lon):
    """
    Estimate proxy environmental features (NDVI, NDWI, elevation, slope,
    distance-to-forest/water/road) for a given point.

    Previously this hard-gated on a fixed "Western Ghats" bounding box:
    inside it, features decayed smoothly with distance from the nearest
    known forest zone; outside it, every single point got the SAME
    hardcoded low-vegetation/urban-like values, regardless of how close
    it actually was to a real conflict zone. That meant the model was
    never shown real geography for anywhere outside the box, so it
    always predicted LOW there.

    Fix: there's no more bounding-box gate, and FOREST_ZONES / URBAN_ZONES
    now cover all of India (Western Ghats, Central India, Himalayan/Terai
    belt, Northeast, Eastern India, Rajasthan/Gujarat), not just Karnataka/
    Kerala/Tamil Nadu. Every location gets a distance-based estimate
    relative to the nearest entry in FOREST_ZONES: close to a known zone
    → high NDVI/elevation, low distance-to-forest; far from all zones →
    decays smoothly toward a baseline "non-forest" profile.

    This is still an approximation (not live satellite data) — accuracy
    depends on how well FOREST_ZONES/URBAN_ZONES represent the real
    forest cover and cities near a given point. Add more entries to
    either list to sharpen predictions in a specific area.
    """
    best_dist = float('inf')
    best_zone = None
    for zone in FOREST_ZONES:
        dist = math.sqrt((lat - zone[0]) ** 2 + (lon - zone[1]) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_zone = zone

    if best_dist <= best_zone[2]:
        # Inside (or blending into) a known forest zone's radius.
        blend = 1.0 - (best_dist / best_zone[2])
        NDVI = best_zone[3] * blend + 0.18 * (1 - blend)
        NDWI = 0.35 * blend + 0.08 * (1 - blend)
        elev = best_zone[4] * blend + 300 * (1 - blend)
        slope = best_zone[5] * blend + 2 * (1 - blend)
        df = best_zone[6] * blend + 9.0 * (1 - blend)
        dw = best_zone[7] * blend + 5.0 * (1 - blend)
        dr = best_zone[8] * blend + 0.4 * (1 - blend)
    else:
        # Beyond every known zone's radius: decay smoothly with distance
        # instead of snapping to a fixed placeholder. Decay reaches its
        # baseline (non-forest) values by ~3 degrees (~330km) away.
        dist_factor = min(1.0, best_dist / 3.0)
        NDVI = 0.60 - dist_factor * 0.42
        NDWI = 0.30 - dist_factor * 0.22
        elev = 600 - dist_factor * 300
        slope = 12 - dist_factor * 10
        df = 1.5 + dist_factor * 7.5
        dw = 0.8 + dist_factor * 4.2
        dr = 2.0 - dist_factor * 1.6

    return NDVI, NDWI, elev, slope, df, dw, dr


def build_features(lat, lon, NDVI, NDWI, elevation, slope,
                    dist_forest, dist_water, dist_road):
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
    best_dist = float('inf')
    best_name = f"({lat:.2f}, {lon:.2f})"
    for plat, plon, name in places:
        dist = math.sqrt((lat - plat) ** 2 + (lon - plon) ** 2)
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
            "risk": "LOW",
            "probability": 5.0,
            "location": urban_name,
            "driver": "human_pressure",
            "lat": lat,
            "lon": lon,
        }

    NDVI, NDWI, elev, slope, df, dw, dr = estimate_features(lat, lon)
    feats = build_features(lat, lon, NDVI, NDWI, elev, slope, df, dw, dr)
    dfx = pd.DataFrame([feats], columns=FEATURES)
    prob = float(model.predict_proba(scaler.transform(dfx))[0][1] * 100)
    risk = "HIGH" if prob >= 70 else "MEDIUM" if prob >= 40 else "LOW"
    location = get_location_name(lat, lon)
    driver = "NDVI" if NDVI > 0.65 else "dist_forest"

    return {
        "risk": risk,
        "probability": round(prob, 2),
        "location": location,
        "driver": driver,
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
    number = int(digest, 16) % 1000000
    return f"{number:06d}"


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
