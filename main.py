import os
import math
import time
import hmac
import hashlib
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# WILDORA HWC BACKEND
# ML + ENVIRONMENT + TIME-AWARE RISK
# ============================================================

app = FastAPI(
    title="HWC Prediction API",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIGURATION
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

MODEL_FILE = "P5model.pkl"
SCALER_FILE = "P5scaler.pkl"
FEATURE_FILE = "P5feature_columns.pkl"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

OVERPASS_HEADERS = {
    "User-Agent": "WILDORA-HWC-App/2.0"
}

NOMINATIM_HEADERS = {
    "User-Agent": "WILDORA-HWC-App/2.0"
}

TERRAIN_CACHE = {}
CACHE_SECONDS = 24 * 60 * 60


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    FEATURES = list(joblib.load(FEATURE_FILE))

    MODEL_LOADED = True

except Exception as e:
    model = None
    scaler = None
    FEATURES = []

    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(e)


# ============================================================
# ORIGINAL 21 FEATURES
# ============================================================

ORIGINAL_FEATURES = [
    "lat",
    "lon",
    "ndvi",
    "ndwi",
    "elevation",
    "slope",
    "dist_forest",
    "dist_water",
    "dist_road",
    "vwr",
    "tri",
    "nld",
    "has",
    "esi",
    "ndvi_ndwi_interaction",
    "veg_water_risk",
    "isolation_index",
    "terrain_ratio",
    "human_pressure",
    "eco_stress",
    "slope_elev_risk",
]


# ============================================================
# REQUEST MODELS
# ============================================================

class PredictionRequest(BaseModel):
    lat: float
    lon: float
    hour: int = 12
    minute: int = 0


class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str


class OTPRequest(BaseModel):
    email: str
    otp: str


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


# ============================================================
# TERRAIN / ENVIRONMENT DATA
# ============================================================

def get_real_terrain(lat, lon):

    cache_key = (
        round(lat, 4),
        round(lon, 4)
    )

    if cache_key in TERRAIN_CACHE:

        saved_time, saved_data = TERRAIN_CACHE[cache_key]

        if time.time() - saved_time < CACHE_SECONDS:
            return saved_data

    try:

        query = f"""
        [out:json][timeout:20];

        (
          way["natural"="forest"](around:10000,{lat},{lon});
          way["landuse"="forest"](around:10000,{lat},{lon});
          relation["landuse"="forest"](around:10000,{lat},{lon});

          way["natural"="water"](around:10000,{lat},{lon});
          way["waterway"](around:10000,{lat},{lon});

          way["highway"](around:5000,{lat},{lon});
        );

        out center;
        """

        response = requests.post(
            OVERPASS_URL,
            data=query,
            headers=OVERPASS_HEADERS,
            timeout=30
        )

        response.raise_for_status()

        elements = response.json().get("elements", [])

        forest_distances = []
        water_distances = []
        road_distances = []

        for element in elements:

            tags = element.get("tags", {})

            center = element.get("center")

            if not center:
                continue

            elat = center.get("lat")
            elon = center.get("lon")

            if elat is None or elon is None:
                continue

            distance = haversine_km(
                lat,
                lon,
                float(elat),
                float(elon)
            )

            if (
                tags.get("natural") == "forest"
                or tags.get("landuse") == "forest"
            ):
                forest_distances.append(distance)

            if (
                tags.get("natural") == "water"
                or "waterway" in tags
            ):
                water_distances.append(distance)

            if "highway" in tags:
                road_distances.append(distance)

        dist_forest = (
            min(forest_distances)
            if forest_distances
            else 8.0
        )

        dist_water = (
            min(water_distances)
            if water_distances
            else 5.0
        )

        dist_road = (
            min(road_distances)
            if road_distances
            else 3.0
        )

        # ----------------------------------------------------
        # ELEVATION
        # ----------------------------------------------------

        elevation = 500.0

        try:

            elev_response = requests.get(
                ELEVATION_URL,
                params={
                    "locations": f"{lat},{lon}"
                },
                timeout=20
            )

            if elev_response.ok:

                result = elev_response.json()

                if result.get("results"):

                    elevation = float(
                        result["results"][0].get(
                            "elevation",
                            500
                        )
                    )

        except Exception:
            pass

        # ----------------------------------------------------
        # ENVIRONMENT ESTIMATION
        # ----------------------------------------------------

        forest_factor = math.exp(
            -dist_forest / 5.0
        )

        water_factor = math.exp(
            -dist_water / 3.0
        )

        road_factor = math.exp(
            -dist_road / 2.0
        )

        ndvi = clamp(
            0.15
            + 0.75 * forest_factor,
            0.05,
            0.95
        )

        ndwi = clamp(
            0.05
            + 0.65 * water_factor,
            -0.20,
            0.90
        )

        # Approximate slope from terrain context.
        slope = clamp(
            5.0
            + abs(math.sin(math.radians(lat))) * 20.0,
            0,
            45
        )

        data = (
            ndvi,
            ndwi,
            elevation,
            slope,
            dist_forest,
            dist_water,
            dist_road
        )

        TERRAIN_CACHE[cache_key] = (
            time.time(),
            data
        )

        return data

    except Exception:

        return None


# ============================================================
# FALLBACK ENVIRONMENT
# ============================================================

def estimate_features_fallback(lat, lon):

    # Smooth geographic/environmental approximation.
    # No named location is hardcoded here.

    lat_factor = abs(
        math.sin(math.radians(lat))
    )

    lon_factor = abs(
        math.cos(math.radians(lon))
    )

    ndvi = clamp(
        0.30 + 0.35 * lat_factor,
        0.10,
        0.85
    )

    ndwi = clamp(
        0.15 + 0.20 * lon_factor,
        -0.10,
        0.70
    )

    elevation = 500 + (
        abs(lat - 12.5) * 40
    )

    slope = clamp(
        8 + lat_factor * 15,
        0,
        35
    )

    dist_forest = 4.0
    dist_water = 3.0
    dist_road = 1.5

    return (
        ndvi,
        ndwi,
        elevation,
        slope,
        dist_forest,
        dist_water,
        dist_road
    )


# ============================================================
# BUILD MODEL FEATURES
# ============================================================

def build_features(
    lat,
    lon,
    ndvi,
    ndwi,
    elevation,
    slope,
    dist_forest,
    dist_water,
    dist_road,
    hour,
    minute
):

    # --------------------------------------------------------
    # Derived environmental variables
    # --------------------------------------------------------

    vwr = (
        ndvi * ndwi
    )

    tri = (
        abs(slope)
        * (1 + elevation / 1000)
    )

    nld = (
        1.0 / (1.0 + dist_forest)
    )

    has = (
        1.0 / (1.0 + dist_water)
    )

    esi = (
        ndvi
        * (1.0 + has)
    )

    ndvi_ndwi_interaction = (
        ndvi * ndwi
    )

    veg_water_risk = (
        ndvi * (1 + ndwi)
    )

    isolation_index = (
        1.0 / (
            1.0
            + dist_forest
            + dist_water
        )
    )

    terrain_ratio = (
        slope / (1 + elevation / 1000)
    )

    human_pressure = (
        1.0 / (1.0 + dist_road)
    )

    eco_stress = (
        ndvi
        + ndwi
        + isolation_index
    ) / 3.0

    slope_elev_risk = (
        slope
        * (1 + elevation / 1000)
    )

    values = {
        "lat": lat,
        "lon": lon,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "elevation": elevation,
        "slope": slope,
        "dist_forest": dist_forest,
        "dist_water": dist_water,
        "dist_road": dist_road,
        "vwr": vwr,
        "tri": tri,
        "nld": nld,
        "has": has,
        "esi": esi,
        "ndvi_ndwi_interaction": ndvi_ndwi_interaction,
        "veg_water_risk": veg_water_risk,
        "isolation_index": isolation_index,
        "terrain_ratio": terrain_ratio,
        "human_pressure": human_pressure,
        "eco_stress": eco_stress,
        "slope_elev_risk": slope_elev_risk,

        # These are only used if the saved model
        # was originally trained with them.
        "hour": hour,
        "minute": minute,
        "time_sin": math.sin(
            2 * math.pi * hour / 24
        ),
        "time_cos": math.cos(
            2 * math.pi * hour / 24
        ),
    }

    row = {}

    for feature in FEATURES:

        if feature in values:
            row[feature] = values[feature]

        else:
            row[feature] = 0.0

    return pd.DataFrame(
        [row],
        columns=FEATURES
    )


# ============================================================
# DETERMINE ENVIRONMENT TYPE
# ============================================================

def calculate_environment_context(
    ndvi,
    ndwi,
    dist_forest,
    dist_water,
    dist_road
):

    # Forest proximity is the strongest environmental signal.
    forest_score = (
        math.exp(-dist_forest / 4.0)
    )

    vegetation_score = clamp(
        ndvi,
        0,
        1
    )

    water_score = clamp(
        ndwi,
        0,
        1
    )

    # High forest proximity + vegetation means
    # stronger wildlife habitat context.
    wildlife_habitat_score = (
        0.55 * forest_score
        + 0.30 * vegetation_score
        + 0.15 * water_score
    )

    wildlife_habitat_score = clamp(
        wildlife_habitat_score,
        0,
        1
    )

    return wildlife_habitat_score


# ============================================================
# TIME-AWARE RISK ADJUSTMENT
# ============================================================

def apply_time_risk(
    base_probability,
    hour,
    minute,
    habitat_score
):

    probability = float(base_probability)

    current_minutes = (
        hour * 60
        + minute
    )

    # --------------------------------------------------------
    # Wildlife activity periods
    # --------------------------------------------------------
    #
    # Evening/night:
    # 18:00 - 06:00
    #
    # Safer daylight:
    # 09:00 - 15:00
    #
    # Transition:
    # 06:00 - 09:00
    # 15:00 - 18:00
    #
    # This is a risk adjustment, NOT a replacement
    # for the ML model.
    # --------------------------------------------------------

    if (
        current_minutes >= 18 * 60
        or current_minutes < 6 * 60
    ):

        # Stronger increase in forest/wildlife habitat.
        adjustment = (
            22.0 * habitat_score
        )

        probability += adjustment

        time_period = "HIGH_WILDLIFE_ACTIVITY"

    elif (
        9 * 60
        <= current_minutes
        < 15 * 60
    ):

        # Daytime reduction is deliberately limited.
        # A genuinely high-risk location remains high.
        adjustment = (
            -8.0 * habitat_score
        )

        probability += adjustment

        time_period = "LOWER_WILDLIFE_ACTIVITY"

    else:

        adjustment = (
            8.0 * habitat_score
        )

        probability += adjustment

        time_period = "TRANSITION"

    probability = clamp(
        probability,
        0,
        100
    )

    return probability, time_period


# ============================================================
# RISK CATEGORY
# ============================================================

def risk_category(probability):

    if probability >= 70:
        return "HIGH"

    if probability >= 40:
        return "MEDIUM"

    return "LOW"


# ============================================================
# PREDICTION DRIVER
# ============================================================

def determine_driver(
    ndvi,
    ndwi,
    dist_forest,
    dist_water,
    dist_road
):

    scores = {
        "forest_proximity": (
            1.0 / (1.0 + dist_forest)
        ),

        "water_proximity": (
            ndwi
            + 1.0 / (1.0 + dist_water)
        ),

        "road_proximity": (
            1.0 / (1.0 + dist_road)
        ),

        "vegetation": ndvi
    }

    return max(
        scores,
        key=scores.get
    )


# ============================================================
# LOCATION NAME
# ============================================================

def get_location_name(lat, lon):

    try:

        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 12
            },
            headers=NOMINATIM_HEADERS,
            timeout=15
        )

        if response.ok:

            data = response.json()

            address = data.get(
                "address",
                {}
            )

            for key in [
                "city",
                "town",
                "village",
                "municipality",
                "county",
                "state_district"
            ]:

                if address.get(key):
                    return address[key]

    except Exception:
        pass

    return "Selected location"


# ============================================================
# MAIN PREDICTION
# ============================================================

def run_prediction(
    lat,
    lon,
    hour,
    minute
):

    if not MODEL_LOADED:

        raise HTTPException(
            status_code=500,
            detail=(
                "ML model could not be loaded: "
                + MODEL_LOAD_ERROR
            )
        )

    if not (
        -90 <= lat <= 90
        and -180 <= lon <= 180
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid latitude or longitude"
        )

    if not 0 <= hour <= 23:

        raise HTTPException(
            status_code=400,
            detail="Hour must be between 0 and 23"
        )

    if not 0 <= minute <= 59:

        raise HTTPException(
            status_code=400,
            detail="Minute must be between 0 and 59"
        )

    # --------------------------------------------------------
    # REAL TERRAIN
    # --------------------------------------------------------

    terrain = get_real_terrain(
        lat,
        lon
    )

    used_fallback = False

    if terrain is None:

        terrain = estimate_features_fallback(
            lat,
            lon
        )

        used_fallback = True

    (
        ndvi,
        ndwi,
        elevation,
        slope,
        dist_forest,
        dist_water,
        dist_road
    ) = terrain

    # --------------------------------------------------------
    # MODEL FEATURES
    # --------------------------------------------------------

    feature_df = build_features(
        lat,
        lon,
        ndvi,
        ndwi,
        elevation,
        slope,
        dist_forest,
        dist_water,
        dist_road,
        hour,
        minute
    )

    try:

        scaled_features = scaler.transform(
            feature_df
        )

        probabilities = model.predict_proba(
            scaled_features
        )[0]

        classes = list(
            model.classes_
        )

        if 1 in classes:

            positive_index = classes.index(1)

        else:

            positive_index = len(
                probabilities
            ) - 1

        base_probability = (
            float(
                probabilities[
                    positive_index
                ]
            )
            * 100
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                + str(e)
            )
        )

    # --------------------------------------------------------
    # ENVIRONMENT CONTEXT
    # --------------------------------------------------------

    habitat_score = calculate_environment_context(
        ndvi,
        ndwi,
        dist_forest,
        dist_water,
        dist_road
    )

    # --------------------------------------------------------
    # TIME-AWARE ADJUSTMENT
    # --------------------------------------------------------

    final_probability, time_period = apply_time_risk(
        base_probability,
        hour,
        minute,
        habitat_score
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # HIGH BASE RISK MUST STAY HIGH DURING DAYTIME
    # --------------------------------------------------------

    if base_probability >= 70:

        final_probability = max(
            final_probability,
            70.0
        )

    risk = risk_category(
        final_probability
    )

    # --------------------------------------------------------
    # DRIVER
    # --------------------------------------------------------

    driver = determine_driver(
        ndvi,
        ndwi,
        dist_forest,
        dist_water,
        dist_road
    )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    location_name = get_location_name(
        lat,
        lon
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "risk": risk,

        "probability": round(
            final_probability,
            2
        ),

        "base_probability": round(
            base_probability,
            2
        ),

        "time_multiplier": 1.0,

        "time_profile": time_period,

        "hour_used": hour,

        "minute_used": minute,

        "time_used": (
            f"{hour:02d}:{minute:02d}"
        ),

        "location": location_name,

        "driver": driver,

        "used_ml_model": True,

        "used_fallback_terrain": used_fallback,

        "lat": lat,

        "lon": lon,

        "model_features": len(FEATURES),

        "habitat_score": round(
            habitat_score,
            3
        ),

        "environment": {
            "ndvi": round(
                ndvi,
                4
            ),

            "ndwi": round(
                ndwi,
                4
            ),

            "elevation": round(
                elevation,
                2
            ),

            "slope": round(
                slope,
                2
            ),

            "distance_forest_km": round(
                dist_forest,
                3
            ),

            "distance_water_km": round(
                dist_water,
                3
            ),

            "distance_road_km": round(
                dist_road,
                3
            )
        }
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "ml_model_loaded": MODEL_LOADED,
        "features": len(FEATURES)
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "WILDORA HWC Prediction API",
        "status": "running",
        "ml_model_loaded": MODEL_LOADED,
        "features": len(FEATURES)
    }


# ============================================================
# GET PREDICTION
# ============================================================

@app.get("/predict")
def predict_get(
    lat: float,
    lon: float,
    hour: int = 12,
    minute: int = 0
):

    return run_prediction(
        lat,
        lon,
        hour,
        minute
    )


# ============================================================
# POST PREDICTION
# ============================================================

@app.post("/predict")
def predict_post(
    request: PredictionRequest
):

    return run_prediction(
        request.lat,
        request.lon,
        request.hour,
        request.minute
    )


# ============================================================
# DATABASE
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

DB_FILE = "wildora.db"


def get_db():

    connection = sqlite3.connect(
        DB_FILE
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_db():

    connection = get_db()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            username TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            verified INTEGER DEFAULT 0,
            created_at TEXT
        )
        """
    )

    connection.commit()

    connection.close()


init_db()


# ============================================================
# PASSWORD HASH
# ============================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ============================================================
# REGISTER
# ============================================================

@app.post("/register-pending")
def register_pending(
    request: RegisterRequest
):

    email = request.email.strip().lower()
    username = request.username.strip()

    if not email or not username or not request.password:

        raise HTTPException(
            status_code=400,
            detail="All fields are required"
        )

    connection = get_db()

    cursor = connection.cursor()

    existing = cursor.execute(
        "SELECT id FROM users WHERE email = ?",
        (email,)
    ).fetchone()

    if existing:

        connection.close()

        raise HTTPException(
            status_code=400,
            detail="Account already exists"
        )

    cursor.execute(
        """
        INSERT INTO users
        (
            email,
            username,
            password_hash,
            verified,
            created_at
        )
        VALUES (?, ?, ?, 0, ?)
        """,
        (
            email,
            username,
            hash_password(
                request.password
            ),
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    connection.commit()

    connection.close()

    return {
        "success": True,
        "message": "Account created successfully"
    }


# ============================================================
# SEND OTP
# ============================================================

OTP_SECRET = os.getenv(
    "OTP_SECRET",
    "WILDORA_DEFAULT_SECRET"
)

OTP_STEP_SECONDS = 300


def generate_otp(email):

    current_step = int(
        time.time()
        // OTP_STEP_SECONDS
    )

    message = (
        f"{email}:{current_step}"
    )

    digest = hmac.new(
        OTP_SECRET.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()

    number = (
        int(digest[:12], 16)
        % 1000000
    )

    return f"{number:06d}"


def send_email_otp(
    email,
    otp
):

    api_key = os.getenv(
        "BREVO_API_KEY"
    )

    sender_email = os.getenv(
        "SENDER_EMAIL"
    )

    if not api_key or not sender_email:

        return False

    payload = {
        "sender": {
            "email": sender_email,
            "name": "WILDORA"
        },

        "to": [
            {
                "email": email
            }
        ],

        "subject": "WILDORA OTP",

        "htmlContent": (
            f"""
            <h2>WILDORA</h2>
            <p>Your OTP is:</p>
            <h1>{otp}</h1>
            <p>This OTP is valid for 5 minutes.</p>
            """
        )
    }

    try:

        response = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={
                "api-key": api_key,
                "Content-Type":
                    "application/json"
            },
            json=payload,
            timeout=20
        )

        return response.ok

    except Exception:

        return False


@app.post("/send-otp")
def send_otp(
    request: OTPRequest
):

    email = request.email.strip().lower()

    otp = generate_otp(
        email
    )

    sent = send_email_otp(
        email,
        otp
    )

    if not sent:

        raise HTTPException(
            status_code=500,
            detail="Unable to send OTP"
        )

    return {
        "success": True,
        "message": "OTP sent successfully"
    }


# ============================================================
# VERIFY OTP
# ============================================================

@app.post("/verify-otp")
def verify_otp(
    request: OTPRequest
):

    email = request.email.strip().lower()

    expected = generate_otp(
        email
    )

    if not hmac.compare_digest(
        request.otp,
        expected
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid or expired OTP"
        )

    connection = get_db()

    cursor = connection.cursor()

    user = cursor.execute(
        """
        SELECT id, email, username
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()

    if not user:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Account not found"
        )

    cursor.execute(
        """
        UPDATE users
        SET verified = 1
        WHERE email = ?
        """,
        (email,)
    )

    connection.commit()

    connection.close()

    return {
        "success": True,
        "message": "OTP verified",
        "email": user["email"],
        "username": user["username"]
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print(
        "======================================"
    )

    print(
        "WILDORA HWC BACKEND STARTED"
    )

    print(
        f"ML MODEL LOADED: {MODEL_LOADED}"
    )

    print(
        f"MODEL FEATURES: {len(FEATURES)}"
    )

    print(
        "RISK LEVELS: HIGH / MEDIUM / LOW"
    )

    print(
        "TIME-AWARE RISK: ENABLED"
    )

    print(
        "======================================"
    )
