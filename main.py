from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import math
import time
import os
import requests
import hmac
import hashlib
import sqlite3

from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# APP
# ============================================================

app = FastAPI(title="HWC Prediction API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD TRAINED ML MODEL
# ============================================================

model = joblib.load("P5model.pkl")
scaler = joblib.load("P5scaler.pkl")
FEATURES = joblib.load("P5feature_columns.pkl")


print("Loaded ML model successfully.")
print("Model features:", FEATURES)


# ============================================================
# OTP CONFIGURATION
# ============================================================

BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
OTP_SECRET = os.environ.get("OTP_SECRET")

OTP_STEP_SECONDS = 300

IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# TERRAIN CACHE
# ============================================================

_terrain_cache = {}

_CACHE_TTL = 24 * 3600


OVERPASS_HEADERS = {
    "User-Agent": "WILDORA-HWC-App/1.0"
}


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):

    r = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * r * math.asin(
        math.sqrt(a)
    )


# ============================================================
# REAL TERRAIN / ENVIRONMENT DATA
#
# No predefined forest list.
# No predefined city list.
# No predefined risk zones.
# ============================================================

def get_real_terrain(lat, lon):

    key = (
        round(lat, 3),
        round(lon, 3)
    )

    cached = _terrain_cache.get(key)

    if cached:

        if (
            time.time() - cached["time"]
            < _CACHE_TTL
        ):

            return cached["data"]


    # --------------------------------------------------------
    # OPENSTREETMAP / OVERPASS
    # --------------------------------------------------------

    query = f"""
    [out:json][timeout:20];

    (
        way["natural"="wood"](around:10000,{lat},{lon});
        way["landuse"="forest"](around:10000,{lat},{lon});

        relation["natural"="wood"](around:10000,{lat},{lon});
        relation["landuse"="forest"](around:10000,{lat},{lon});

        way["natural"="water"](around:10000,{lat},{lon});
        way["waterway"](around:10000,{lat},{lon});

        way["highway"](around:5000,{lat},{lon});
    );

    out center;
    """

    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        headers=OVERPASS_HEADERS,
        timeout=25
    )

    response.raise_for_status()

    elements = response.json().get(
        "elements",
        []
    )


    dist_forest = None
    dist_water = None
    dist_road = None


    for element in elements:

        center = element.get("center")

        if not center:
            continue

        element_lat = center.get("lat")
        element_lon = center.get("lon")

        if (
            element_lat is None
            or element_lon is None
        ):
            continue


        distance = haversine_km(
            lat,
            lon,
            element_lat,
            element_lon
        )


        tags = element.get(
            "tags",
            {}
        )


        # FOREST

        if (
            tags.get("natural") == "wood"
            or
            tags.get("landuse") == "forest"
        ):

            if dist_forest is None:

                dist_forest = distance

            else:

                dist_forest = min(
                    dist_forest,
                    distance
                )


        # WATER

        elif (
            tags.get("natural") == "water"
            or
            "waterway" in tags
        ):

            if dist_water is None:

                dist_water = distance

            else:

                dist_water = min(
                    dist_water,
                    distance
                )


        # ROAD

        elif "highway" in tags:

            if dist_road is None:

                dist_road = distance

            else:

                dist_road = min(
                    dist_road,
                    distance
                )


    # --------------------------------------------------------
    # SAFE FALLBACKS
    #
    # These are only missing-data fallbacks.
    # They do NOT determine the risk directly.
    # --------------------------------------------------------

    if dist_forest is None:
        dist_forest = 10.0

    if dist_water is None:
        dist_water = 5.0

    if dist_road is None:
        dist_road = 1.0


    # --------------------------------------------------------
    # ELEVATION
    # --------------------------------------------------------

    elevation = 500.0
    slope = 5.0


    try:

        elevation_response = requests.post(

            "https://api.open-elevation.com/api/v1/lookup",

            json={
                "locations": [
                    {
                        "latitude": lat,
                        "longitude": lon
                    },
                    {
                        "latitude": lat + 0.003,
                        "longitude": lon
                    },
                    {
                        "latitude": lat,
                        "longitude": lon + 0.003
                    }
                ]
            },

            timeout=20
        )


        elevation_response.raise_for_status()

        results = elevation_response.json().get(
            "results",
            []
        )


        if len(results) >= 3:

            elevation = float(
                results[0]["elevation"]
            )

            elevation_lat = float(
                results[1]["elevation"]
            )

            elevation_lon = float(
                results[2]["elevation"]
            )


            dz_lat = abs(
                elevation_lat - elevation
            )

            dz_lon = abs(
                elevation_lon - elevation
            )


            slope = min(
                45.0,
                (
                    max(
                        dz_lat,
                        dz_lon
                    ) / 300.0
                ) * 100
            )


    except Exception as error:

        print(
            "Elevation lookup failed:",
            error
        )


    # --------------------------------------------------------
    # ENVIRONMENTAL INDICES
    #
    # These are continuous feature estimates used only when
    # satellite NDVI/NDWI is not directly available.
    # --------------------------------------------------------

    ndvi = max(
        0.05,
        min(
            0.90,
            0.85 - (
                dist_forest * 0.08
            )
        )
    )


    ndwi = max(
        0.02,
        min(
            0.40,
            0.40 - (
                dist_water * 0.05
            )
        )
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


    _terrain_cache[key] = {
        "time": time.time(),
        "data": data
    }


    return data


# ============================================================
# GENERIC FALLBACK
#
# No city names.
# No forest names.
# No risk zones.
# ============================================================

def estimate_features_fallback(lat, lon):

    # Smooth geographic/environmental fallback.
    # It is only used if external terrain services fail.

    latitude_factor = (
        abs(lat - 20.0) / 20.0
    )

    longitude_factor = (
        abs(lon - 78.0) / 15.0
    )


    latitude_factor = min(
        1.0,
        latitude_factor
    )

    longitude_factor = min(
        1.0,
        longitude_factor
    )


    environment_factor = (
        1.0
        -
        (
            latitude_factor
            +
            longitude_factor
        ) / 2
    )


    ndvi = (
        0.25
        +
        0.45 * environment_factor
    )


    ndwi = (
        0.10
        +
        0.20 * environment_factor
    )


    elevation = (
        300
        +
        500 * environment_factor
    )


    slope = (
        3
        +
        15 * environment_factor
    )


    dist_forest = (
        2
        +
        6 * (
            1 - environment_factor
        )
    )


    dist_water = (
        1
        +
        4 * (
            1 - environment_factor
        )
    )


    dist_road = (
        0.3
        +
        2 * environment_factor
    )


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
# FEATURE ENGINEERING
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
    hour=None,
    minute=None
):

    vwr = (
        ndvi /
        (ndwi + 0.01)
    )


    tri = slope * 0.5


    nld = min(
        1,
        max(
            0,
            (lon - 75) / 5
        )
    )


    has = min(
        1,
        max(
            0,
            (lat - 10) / 5
        )
    )


    esi = (
        ndvi + ndwi
    ) / 2


    ndvi_ndwi_interaction = (
        ndvi * ndwi
    )


    veg_water_risk = (
        ndvi /
        (dist_water + 0.1)
    )


    isolation_index = (
        dist_forest +
        dist_road
    ) / 2


    terrain_ratio = (
        slope /
        (elevation + 1)
    )


    human_pressure = (
        has + nld
    ) / 2


    eco_stress = (
        ndvi +
        ndwi +
        has
    ) / 3


    slope_elev_risk = (
        slope * elevation
    ) / 1000


    # --------------------------------------------------------
    # EXISTING 21 FEATURES
    # --------------------------------------------------------

    feature_values = {

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

        "ndvi_ndwi_interaction":
            ndvi_ndwi_interaction,

        "veg_water_risk":
            veg_water_risk,

        "isolation_index":
            isolation_index,

        "terrain_ratio":
            terrain_ratio,

        "human_pressure":
            human_pressure,

        "eco_stress":
            eco_stress,

        "slope_elev_risk":
            slope_elev_risk,
    }


    # --------------------------------------------------------
    # IF THE MODEL WAS TRAINED WITH TIME FEATURES,
    # SUPPORT THEM AUTOMATICALLY.
    #
    # This does NOT add new columns to an old model.
    # --------------------------------------------------------

    if hour is not None:

        hour_value = int(hour)

        minute_value = (
            0
            if minute is None
            else int(minute)
        )


        decimal_hour = (
            hour_value
            +
            minute_value / 60.0
        )


        time_angle = (
            2 * math.pi
            * decimal_hour
            / 24.0
        )


        feature_values["hour"] = (
            hour_value
        )

        feature_values["minute"] = (
            minute_value
        )

        feature_values["time_sin"] = (
            math.sin(time_angle)
        )

        feature_values["time_cos"] = (
            math.cos(time_angle)
        )


    # --------------------------------------------------------
    # RETURN EXACTLY THE FEATURES EXPECTED BY THE MODEL
    # --------------------------------------------------------

    final_values = []

    for feature in FEATURES:

        if feature in feature_values:

            final_values.append(
                feature_values[feature]
            )

        else:

            # Unknown feature from the saved model.
            # Use zero rather than changing model dimensions.
            final_values.append(0.0)


    return final_values


# ============================================================
# LOCATION NAME
#
# Uses reverse geocoding instead of predefined locations.
# ============================================================

def get_location_name(lat, lon):

    try:

        response = requests.get(

            "https://nominatim.openstreetmap.org/reverse",

            params={
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 10
            },

            headers={
                "User-Agent":
                    "WILDORA-HWC-App/1.0"
            },

            timeout=10
        )


        response.raise_for_status()

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
            "state"
        ]:

            if address.get(key):

                return address[key]


    except Exception as error:

        print(
            "Reverse geocoding failed:",
            error
        )


    return (
        f"({lat:.4f}, {lon:.4f})"
    )


# ============================================================
# DRIVER
# ============================================================

def determine_driver(
    ndvi,
    ndwi,
    dist_forest,
    dist_water,
    dist_road
):

    # This describes the strongest environmental feature.
    # It does NOT override the ML result.

    scores = {

        "vegetation":
            ndvi,

        "forest_proximity":
            1 / (
                dist_forest + 0.1
            ),

        "water_proximity":
            1 / (
                dist_water + 0.1
            ),

        "road_proximity":
            1 / (
                dist_road + 0.1
            )
    }


    return max(
        scores,
        key=scores.get
    )


# ============================================================
# MAIN ML PREDICTION
# ============================================================

def run_prediction(
    lat: float,
    lon: float,
    hour: Optional[int] = None,
    minute: Optional[int] = None
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not (
        -90 <= lat <= 90
    ):

        raise ValueError(
            "Invalid latitude"
        )


    if not (
        -180 <= lon <= 180
    ):

        raise ValueError(
            "Invalid longitude"
        )


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    now = datetime.now(IST)


    hour_used = (
        now.hour
        if hour is None
        else int(hour)
    )


    minute_used = (
        now.minute
        if minute is None
        else int(minute)
    )


    if not (
        0 <= hour_used <= 23
    ):

        raise ValueError(
            "Hour must be between 0 and 23"
        )


    if not (
        0 <= minute_used <= 59
    ):

        raise ValueError(
            "Minute must be between 0 and 59"
        )


    # --------------------------------------------------------
    # TERRAIN / ENVIRONMENT
    # --------------------------------------------------------

    used_fallback = False


    try:

        (
            ndvi,
            ndwi,
            elevation,
            slope,
            dist_forest,
            dist_water,
            dist_road

        ) = get_real_terrain(
            lat,
            lon
        )


    except Exception as error:

        print(
            "Terrain lookup failed:",
            error
        )


        (
            ndvi,
            ndwi,
            elevation,
            slope,
            dist_forest,
            dist_water,
            dist_road

        ) = estimate_features_fallback(
            lat,
            lon
        )


        used_fallback = True


    # --------------------------------------------------------
    # BUILD MODEL FEATURES
    # --------------------------------------------------------

    features = build_features(

        lat,

        lon,

        ndvi,

        ndwi,

        elevation,

        slope,

        dist_forest,

        dist_water,

        dist_road,

        hour_used,

        minute_used
    )


    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    dataframe = pd.DataFrame(
        [features],
        columns=FEATURES
    )


    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    scaled = scaler.transform(
        dataframe
    )


    # --------------------------------------------------------
    # ML MODEL
    # --------------------------------------------------------

    prediction_probabilities = (
        model.predict_proba(
            scaled
        )[0]
    )


    # --------------------------------------------------------
    # FIND POSITIVE CLASS
    #
    # Normally class 1 = conflict.
    # --------------------------------------------------------

    classes = list(
        getattr(
            model,
            "classes_",
            [0, 1]
        )
    )


    if 1 in classes:

        positive_index = (
            classes.index(1)
        )

    else:

        positive_index = (
            len(classes) - 1
        )


    base_probability = float(
        prediction_probabilities[
            positive_index
        ] * 100
    )


    base_probability = max(
        0,
        min(
            100,
            base_probability
        )
    )


    # --------------------------------------------------------
    # RISK CATEGORY
    #
    # IMPORTANT:
    # This comes directly from ML probability.
    #
    # No urban bypass.
    # No forest-zone bypass.
    # No time multiplier.
    # No predefined location risk.
    # --------------------------------------------------------

    if base_probability >= 70:

        risk = "HIGH"

    elif base_probability >= 40:

        risk = "MEDIUM"

    else:

        risk = "LOW"


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
    # LOCATION NAME
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
            base_probability,
            2
        ),

        "base_probability": round(
            base_probability,
            2
        ),

        "time_multiplier": 1.0,

        "time_profile":
            "ML_MODEL",

        "hour_used":
            hour_used,

        "minute_used":
            minute_used,

        "time_used":
            f"{hour_used:02d}:{minute_used:02d}",

        "location":
            location_name,

        "driver":
            driver,

        "used_ml_model":
            True,

        "used_fallback_terrain":
            used_fallback,

        "lat":
            lat,

        "lon":
            lon,

        "model_features":
            len(FEATURES)
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "ml_model_loaded": True,
        "features": len(FEATURES)
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "name":
            "HWC Prediction API",

        "status":
            "live",

        "prediction_engine":
            "Machine Learning Model"
    }


# ============================================================
# GET PREDICTION
# ============================================================

@app.get("/predict")
def predict_get(

    lat: float,

    lon: float,

    hour: Optional[int] = None,

    minute: Optional[int] = None

):

    return run_prediction(

        lat,

        lon,

        hour,

        minute
    )


# ============================================================
# POST REQUEST
# ============================================================

class PredictRequest(BaseModel):

    lat: float

    lon: float

    hour: Optional[int] = None

    minute: Optional[int] = None


@app.post("/predict")
def predict_post(
    req: PredictRequest
):

    return run_prediction(

        req.lat,

        req.lon,

        req.hour,

        req.minute
    )


# ============================================================
# ACCOUNT / OTP MODELS
# ============================================================

class EmailRequest(BaseModel):

    email: str

    purpose: str

    username: Optional[str] = None


class VerifyRequest(BaseModel):

    email: str

    otp: str

    purpose: str

    username: Optional[str] = None


# ============================================================
# DATABASE
# ============================================================

DATABASE_URL = os.environ.get(
    "DATABASE_URL"
)


def _db_connection():

    if DATABASE_URL:

        try:

            import psycopg2

            return psycopg2.connect(
                DATABASE_URL
            )

        except Exception as error:

            raise RuntimeError(
                f"Could not connect to PostgreSQL: {error}"
            )


    connection = sqlite3.connect(
        "accounts.db",
        timeout=30
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


def _init_accounts_db():

    connection = _db_connection()

    try:

        cursor = connection.cursor()


        if DATABASE_URL:

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS accounts (

                    email TEXT PRIMARY KEY,

                    username TEXT NOT NULL,

                    created_at TIMESTAMP
                    NOT NULL
                    DEFAULT CURRENT_TIMESTAMP

                )

            """)

        else:

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS accounts (

                    email TEXT PRIMARY KEY,

                    username TEXT NOT NULL,

                    created_at TEXT
                    NOT NULL
                    DEFAULT CURRENT_TIMESTAMP

                )

            """)


        connection.commit()


    finally:

        connection.close()


_init_accounts_db()


# ============================================================
# GET ACCOUNT
# ============================================================

def get_account(email: str):

    email = email.lower().strip()


    connection = _db_connection()


    try:

        cursor = connection.cursor()


        cursor.execute(

            "SELECT email, username "
            "FROM accounts "
            "WHERE email = %s"

            if DATABASE_URL

            else

            "SELECT email, username "
            "FROM accounts "
            "WHERE email = ?",

            (email,)

        )


        row = cursor.fetchone()


        if row is None:

            return None


        if DATABASE_URL:

            return {

                "email":
                    row[0],

                "username":
                    row[1]

            }


        return {

            "email":
                row["email"],

            "username":
                row["username"]

        }


    finally:

        connection.close()


# ============================================================
# CREATE ACCOUNT
# ============================================================

def create_account(
    email: str,
    username: str
):

    email = email.lower().strip()

    username = username.strip()


    connection = _db_connection()


    try:

        cursor = connection.cursor()


        try:

            cursor.execute(

                "INSERT INTO accounts "
                "(email, username) "
                "VALUES (%s, %s)"

                if DATABASE_URL

                else

                "INSERT INTO accounts "
                "(email, username) "
                "VALUES (?, ?)",

                (
                    email,
                    username
                )

            )


            connection.commit()

            return True


        except Exception:

            connection.rollback()

            return False


    finally:

        connection.close()


# ============================================================
# DELETE ACCOUNT
# ============================================================

def delete_account(email: str):

    email = email.lower().strip()


    connection = _db_connection()


    try:

        cursor = connection.cursor()


        cursor.execute(

            "DELETE FROM accounts "
            "WHERE email = %s"

            if DATABASE_URL

            else

            "DELETE FROM accounts "
            "WHERE email = ?",

            (email,)

        )


        connection.commit()


        return cursor.rowcount > 0


    finally:

        connection.close()


# ============================================================
# ACCOUNT STATUS
# ============================================================

@app.get("/account-status")
def account_status(email: str):

    email = email.lower().strip()


    if not email:

        return {
            "error":
                "Email is required"
        }


    try:

        account = get_account(
            email
        )


    except Exception as error:

        return {
            "error":
                f"Account database error: {error}"
        }


    if account:

        return {

            "exists":
                True,

            "account":
                account

        }


    return {
        "exists":
            False
    }


# ============================================================
# REMOVE ACCOUNT
# ============================================================

@app.delete("/account")
def remove_account(email: str):

    email = email.lower().strip()


    if not email:

        return {
            "error":
                "Email is required"
        }


    try:

        deleted = delete_account(
            email
        )


    except Exception as error:

        return {

            "error":
                f"Account database error: {error}"

        }


    return {
        "deleted":
            deleted
    }


# ============================================================
# OTP PURPOSE
# ============================================================

def normalize_otp_purpose(
    purpose: str
):

    purpose = purpose.lower().strip()


    if purpose not in (
        "create",
        "login"
    ):

        raise ValueError(
            "Invalid OTP purpose"
        )


    return purpose


# ============================================================
# SEND EMAIL OTP
# ============================================================

def send_email_otp(

    to_email: str,

    otp: str,

    purpose: str

):

    purpose = normalize_otp_purpose(
        purpose
    )


    if purpose == "create":

        subject = (
            "WILDORA - Verify Your Email"
        )


        message = (

            f"Your WILDORA account "
            f"verification code is: {otp}\n\n"

            "Use this code to verify "
            "your email and create your account.\n\n"

            "Expires in 5 minutes."

        )


    else:

        subject = (
            "WILDORA - Login Code"
        )


        message = (

            f"Your WILDORA login "
            f"code is: {otp}\n\n"

            "Use this code to sign in "
            "to your account.\n\n"

            "Expires in 5 minutes."

        )


    response = requests.post(

        "https://api.brevo.com/v3/smtp/email",

        headers={

            "api-key":
                BREVO_API_KEY,

            "Content-Type":
                "application/json",

            "Accept":
                "application/json"

        },

        json={

            "sender": {

                "email":
                    SENDER_EMAIL,

                "name":
                    "WILDORA"

            },

            "to": [

                {
                    "email":
                        to_email
                }

            ],

            "subject":
                subject,

            "textContent":
                message

        },

        timeout=15

    )


    if response.status_code >= 300:

        raise Exception(

            f"Brevo API error "
            f"{response.status_code}: "
            f"{response.text}"

        )


# ============================================================
# GENERATE OTP
# ============================================================

def generate_otp(

    email: str,

    purpose: str,

    time_step: int

):

    purpose = normalize_otp_purpose(
        purpose
    )


    msg = (

        f"{email.lower().strip()}:"
        f"{purpose}:"
        f"{time_step}"

    ).encode()


    digest = hmac.new(

        OTP_SECRET.encode(),

        msg,

        hashlib.sha256

    ).hexdigest()


    return (
        f"{int(digest, 16) % 1000000:06d}"
    )


# ============================================================
# SEND OTP
# ============================================================

@app.post("/send-otp")
def send_otp(
    req: EmailRequest
):

    try:

        purpose = (
            normalize_otp_purpose(
                req.purpose
            )
        )

    except ValueError:

        return {
            "error":
                "Invalid OTP purpose"
        }


    email = (
        req.email.lower().strip()
    )


    if not email:

        return {
            "error":
                "Email is required"
        }


    # CREATE

    if purpose == "create":

        try:

            if get_account(email):

                return {

                    "error":
                        "An account already exists "
                        "for this email. Please log in."

                }


        except Exception as error:

            return {

                "error":
                    f"Account database error: {error}"

            }


    # LOGIN

    if purpose == "login":

        try:

            if not get_account(email):

                return {

                    "error":
                        "No WILDORA account found "
                        "for this email. "
                        "Please create an account first."

                }


        except Exception as error:

            return {

                "error":
                    f"Account database error: {error}"

            }


    time_step = int(
        time.time()
        //
        OTP_STEP_SECONDS
    )


    otp = generate_otp(

        email,

        purpose,

        time_step

    )


    try:

        send_email_otp(

            email,

            otp,

            purpose

        )


    except Exception as error:

        return {

            "error":
                f"Failed to send email: {error}"

        }


    return {

        "message":
            "OTP sent",

        "purpose":
            purpose

    }


# ============================================================
# VERIFY OTP
# ============================================================

@app.post("/verify-otp")
def verify_otp(
    req: VerifyRequest
):

    try:

        purpose = (
            normalize_otp_purpose(
                req.purpose
            )
        )

    except ValueError:

        return {

            "error":
                "Invalid OTP purpose"

        }


    email = (
        req.email.lower().strip()
    )


    entered = req.otp.strip()


    if (
        len(entered) != 6
        or
        not entered.isdigit()
    ):

        return {

            "error":
                "OTP must be 6 digits"

        }


    current_step = int(
        time.time()
        //
        OTP_STEP_SECONDS
    )


    current_otp = generate_otp(

        email,

        purpose,

        current_step

    )


    previous_otp = generate_otp(

        email,

        purpose,

        current_step - 1

    )


    if not (

        hmac.compare_digest(
            current_otp,
            entered
        )

        or

        hmac.compare_digest(
            previous_otp,
            entered
        )

    ):

        return {

            "error":
                "Incorrect or expired OTP"

        }


    # CREATE ACCOUNT

    if purpose == "create":

        username = (
            req.username or ""
        ).strip()


        if len(username) < 2:

            return {

                "error":
                    "Username is required "
                    "to create an account"

            }


        try:

            if get_account(email):

                return {

                    "error":
                        "An account already exists "
                        "for this email. "
                        "Please log in."

                }


            if not create_account(
                email,
                username
            ):

                return {

                    "error":
                        "An account already exists "
                        "for this email. "
                        "Please log in."

                }


        except Exception as error:

            return {

                "error":
                    f"Could not save account: {error}"

            }


        return {

            "message":
                "Account created",

            "purpose":
                purpose,

            "account": {

                "email":
                    email,

                "username":
                    username

            }

        }


    # LOGIN

    try:

        account = get_account(
            email
        )


    except Exception as error:

        return {

            "error":
                f"Account database error: {error}"

        }


    if not account:

        return {

            "error":
                "Account does not exist. "
                "Please create an account first."

        }


    return {

        "message":
            "Verified",

        "purpose":
            purpose,

        "account":
            account

    }
