from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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
import json
import re
import secrets
from urllib.parse import quote
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth
from zoneinfo import ZoneInfo

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
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
OTP_SECRET = os.environ.get("OTP_SECRET")
OTP_STEP_SECONDS = 300

# Firebase Admin is used only on the backend. The Firebase account is
# created after the user clicks the email verification link.
FIREBASE_SERVICE_ACCOUNT_JSON = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
VERIFICATION_BASE_URL = os.environ.get(
    "VERIFICATION_BASE_URL",
    "https://hwc-backend-fixed.onrender.com"
).rstrip("/")
VERIFICATION_TTL_SECONDS = 30 * 60

if not FIREBASE_SERVICE_ACCOUNT_JSON:
    raise RuntimeError(
        "FIREBASE_SERVICE_ACCOUNT_JSON environment variable is required"
    )

if not firebase_admin._apps:
    firebase_admin.initialize_app(
        credentials.Certificate(
            json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
        )
    )

db = firestore.client()

IST = ZoneInfo("Asia/Kolkata")


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


FOREST_ZONES = [
    (11.9, 76.1, 1.0, 0.82, 900, 12, 0.2, 0.4, 1.5, "Nagarhole", "elephant"),
    (11.6, 76.4, 1.0, 0.80, 860, 15, 0.2, 0.3, 1.8, "Bandipur", "elephant"),
    (11.6, 76.1, 1.0, 0.85, 790, 18, 0.1, 0.2, 1.6, "Wayanad", "elephant"),
    (12.4, 75.7, 1.0, 0.84, 1000, 22, 0.2, 0.3, 2.0, "Kodagu", "elephant"),
    (13.1, 75.3, 0.8, 0.86, 1100, 26, 0.2, 0.3, 2.5, "Kudremukh", "carnivore"),
    (11.9, 77.0, 0.8, 0.76, 1050, 20, 0.3, 0.5, 2.0, "BRT Hills", "elephant"),
    (11.5, 77.2, 0.8, 0.74, 820, 16, 0.3, 0.5, 2.2, "Sathyamangalam", "elephant"),
    (10.5, 76.9, 0.8, 0.80, 880, 17, 0.2, 0.4, 2.0, "Anamalai", "elephant"),
    (11.4, 76.7, 0.8, 0.78, 1100, 24, 0.2, 0.3, 2.5, "Nilgiris", "elephant"),
    (13.5, 75.7, 0.8, 0.77, 830, 14, 0.2, 0.4, 2.0, "Bhadra", "carnivore"),
    (13.3, 75.8, 0.7, 0.80, 900, 16, 0.2, 0.3, 2.0, "Chikmagalur Forest", "carnivore"),
    (12.6, 75.7, 0.7, 0.83, 1000, 22, 0.2, 0.3, 2.0, "Pushpagiri", "carnivore"),
    (14.0, 74.8, 0.7, 0.80, 680, 16, 0.2, 0.2, 2.5, "Sharavathi", "carnivore"),
    (13.4, 75.1, 0.7, 0.85, 820, 22, 0.2, 0.3, 2.5, "Agumbe", "carnivore"),
    (14.6, 74.8, 0.7, 0.78, 600, 18, 0.3, 0.4, 3.0, "Sirsi", "carnivore"),
    (11.2, 77.5, 0.7, 0.79, 940, 20, 0.2, 0.3, 2.5, "Kalakad", "elephant"),
    (12.0, 75.5, 0.8, 0.80, 850, 18, 0.2, 0.3, 2.0, "Coorg Buffer", "elephant"),
    (12.5, 76.0, 0.7, 0.81, 870, 19, 0.2, 0.3, 2.0, "Kabini", "elephant"),
    (12.2, 75.9, 0.7, 0.82, 880, 20, 0.2, 0.3, 2.0, "Brahmagiri", "elephant"),
    (11.0, 76.5, 0.7, 0.78, 820, 16, 0.2, 0.4, 2.0, "Palakkad Gap", "elephant"),
    (10.8, 76.7, 0.7, 0.76, 750, 14, 0.3, 0.4, 2.0, "Silent Valley", "carnivore"),
    (15.2, 74.6, 0.7, 0.79, 580, 17, 0.3, 0.4, 2.5, "Dandeli", "carnivore"),
    (12.4, 76.0, 0.9, 0.82, 860, 18, 0.2, 0.3, 2.0, "Namdroling Area", "elephant"),
]


NAMED_TIME_ZONES = [
    (28.62, 79.80, 0.7, "diurnal_worker", "Pilibhit Tiger Reserve", 0.72, 150, 3, 0.3, 2.0, 1.0),
    (29.53, 78.77, 0.8, "mixed", "Corbett", 0.78, 600, 15, 0.2, 1.5, 1.5),
    (28.52, 80.60, 0.7, "mixed", "Dudhwa", 0.74, 150, 3, 0.2, 2.0, 1.0),
    (26.58, 93.17, 0.8, "elephant", "Kaziranga", 0.70, 60, 1, 0.3, 0.5, 2.0),
    (22.33, 80.63, 0.7, "carnivore", "Kanha", 0.76, 600, 8, 0.2, 3.0, 2.0),
    (23.68, 80.95, 0.6, "carnivore", "Bandhavgarh", 0.74, 800, 12, 0.2, 3.0, 2.0),
    (26.02, 76.50, 0.6, "carnivore", "Ranthambore", 0.55, 400, 10, 0.3, 2.0, 2.0),
    (21.60, 86.30, 0.7, "mixed", "Similipal", 0.78, 600, 12, 0.2, 3.0, 3.0),
    (21.90, 88.90, 0.8, "diurnal_worker", "Sundarbans", 0.68, 2, 0.5, 0.1, 0.2, 5.0),
    (21.13, 70.80, 0.6, "carnivore", "Gir", 0.60, 300, 8, 0.2, 3.0, 2.0),
    (9.46, 77.24, 0.6, "elephant", "Periyar", 0.82, 900, 20, 0.1, 1.0, 2.0),
    (26.72, 90.98, 0.7, "elephant", "Manas", 0.76, 150, 5, 0.2, 1.0, 2.0),
    (20.23, 79.40, 0.8, "carnivore", "Tadoba-Andhari (Chandrapur)", 0.72, 300, 8, 0.2, 2.5, 1.5),
    (21.33, 77.20, 0.7, "carnivore", "Melghat", 0.75, 500, 12, 0.2, 3.0, 2.0),
]


REGION_PROFILES = [
    (8.0, 16.0, 74.0, 77.5, "elephant"),
    (18.0, 24.0, 77.0, 83.0, "carnivore"),
    (26.0, 30.0, 77.0, 89.0, "mixed"),
    (21.0, 23.0, 88.0, 90.5, "diurnal_worker"),
    (24.0, 29.0, 89.0, 97.0, "carnivore"),
]


TIME_PROFILES = {
    "elephant": [
        0.80, 0.70, 0.70, 0.75, 0.90, 1.30,
        1.45, 1.35, 1.00, 0.80, 0.70, 0.65,
        0.60, 0.65, 0.70, 0.80, 0.95, 1.20,
        1.45, 1.40, 1.20, 1.00, 0.90, 0.85
    ],

    "carnivore": [
        1.30, 1.40, 1.45, 1.35, 1.20, 1.00,
        0.85, 0.75, 0.70, 0.65, 0.60, 0.60,
        0.60, 0.60, 0.65, 0.70, 0.80, 0.90,
        1.00, 1.10, 1.20, 1.25, 1.35, 1.40
    ],

    "diurnal_worker": [
        0.60, 0.55, 0.55, 0.60, 0.70, 0.90,
        1.10, 1.30, 1.40, 1.35, 1.25, 1.15,
        1.05, 1.10, 1.20, 1.25, 1.15, 1.00,
        0.85, 0.70, 0.65, 0.60, 0.58, 0.58
    ],

    "mixed": [
        0.95, 0.90, 0.90, 0.92, 0.95, 1.05,
        1.10, 1.15, 1.05, 1.00, 0.95, 0.92,
        0.90, 0.92, 0.95, 1.00, 1.05, 1.10,
        1.15, 1.10, 1.05, 1.00, 0.97, 0.95
    ],
}


_terrain_cache = {}
_CACHE_TTL = 24 * 3600

# Overpass usage policy asks for an identifying User-Agent on every
# request. Missing this makes requests more likely to be deprioritized
# or rejected by the public instance, which was silently causing
# get_real_terrain() to fail and fall back to generic values for most
# of India.
OVERPASS_HEADERS = {
    "User-Agent": "HWCAlertApp/1.0 (contact: your_email@example.com)"
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * r * math.asin(math.sqrt(a))


def is_urban(lat, lon):
    for ulat, ulon, radius, name in URBAN_ZONES:
        if haversine_km(lat, lon, ulat, ulon) <= radius * 111:
            return True, name

    return False, None


def get_time_profile(lat, lon):

    best_dist = float("inf")
    best_profile = "mixed"

    for zone in FOREST_ZONES:

        distance = math.sqrt(
            (lat - zone[0]) ** 2 +
            (lon - zone[1]) ** 2
        )

        if distance < best_dist:
            best_dist = distance
            best_profile = zone[10]

    if best_dist <= 0.5:
        return best_profile

    for zone in NAMED_TIME_ZONES:

        distance = math.sqrt(
            (lat - zone[0]) ** 2 +
            (lon - zone[1]) ** 2
        )

        if distance <= zone[2]:
            return zone[3]

    for min_lat, max_lat, min_lon, max_lon, profile in REGION_PROFILES:

        if (
            min_lat <= lat <= max_lat
            and min_lon <= lon <= max_lon
        ):
            return profile

    return "mixed"


def get_exact_time_multiplier(
    lat,
    lon,
    hour,
    minute
):

    profile = get_time_profile(
        lat,
        lon
    )

    values = TIME_PROFILES[profile]

    hour = max(
        0,
        min(23, int(hour))
    )

    minute = max(
        0,
        min(59, int(minute))
    )

    current_value = values[hour]

    next_hour = (
        hour + 1
    ) % 24

    next_value = values[next_hour]

    fraction = minute / 60.0

    multiplier = (
        current_value
        + (next_value - current_value)
        * fraction
    )

    return profile, multiplier


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

    # NOTE: headers added — see OVERPASS_HEADERS comment above. This was
    # missing before and is the most likely reason live terrain lookups
    # were failing (silently falling back to generic values for most
    # locations outside the curated zone lists).
    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        headers=OVERPASS_HEADERS,
        timeout=20
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

        distance = haversine_km(
            lat,
            lon,
            center["lat"],
            center["lon"]
        )

        tags = element.get(
            "tags",
            {}
        )

        if (
            tags.get("natural") == "wood"
            or tags.get("landuse") == "forest"
        ):

            if dist_forest is None:
                dist_forest = distance
            else:
                dist_forest = min(
                    dist_forest,
                    distance
                )

        elif (
            tags.get("natural") == "water"
            or "waterway" in tags
        ):

            if dist_water is None:
                dist_water = distance
            else:
                dist_water = min(
                    dist_water,
                    distance
                )

        elif "highway" in tags:

            if dist_road is None:
                dist_road = distance
            else:
                dist_road = min(
                    dist_road,
                    distance
                )

    if dist_forest is None:
        dist_forest = 9.0

    if dist_water is None:
        dist_water = 5.0

    if dist_road is None:
        dist_road = 0.4

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
        timeout=15
    )

    elevation_response.raise_for_status()

    results = elevation_response.json()["results"]

    elevation = results[0]["elevation"]

    dz_lat = abs(
        results[1]["elevation"]
        - elevation
    )

    dz_lon = abs(
        results[2]["elevation"]
        - elevation
    )

    slope = min(
        45.0,
        (max(dz_lat, dz_lon) / 300.0) * 100
    )

    ndvi = max(
        0.15,
        min(
            0.88,
            0.85 - dist_forest * 0.08
        )
    )

    ndwi = max(
        0.05,
        min(
            0.35,
            0.35 - dist_water * 0.05
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


def estimate_features_fallback(lat, lon):

    best_dist = float("inf")
    best_zone = None

    for zone in FOREST_ZONES:

        distance = math.sqrt(
            (lat - zone[0]) ** 2 +
            (lon - zone[1]) ** 2
        )

        if distance < best_dist:
            best_dist = distance
            best_zone = zone

    if (
        best_zone is not None
        and best_dist <= best_zone[2]
    ):

        blend = 1 - (
            best_dist / best_zone[2]
        )

        ndvi = (
            best_zone[3] * blend
            + 0.18 * (1 - blend)
        )

        ndwi = (
            0.35 * blend
            + 0.08 * (1 - blend)
        )

        elevation = (
            best_zone[4] * blend
            + 300 * (1 - blend)
        )

        slope = (
            best_zone[5] * blend
            + 2 * (1 - blend)
        )

        dist_forest = (
            best_zone[6] * blend
            + 9 * (1 - blend)
        )

        dist_water = (
            best_zone[7] * blend
            + 5 * (1 - blend)
        )

        dist_road = (
            best_zone[8] * blend
            + 0.4 * (1 - blend)
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

    best_dist = float("inf")
    best_zone = None

    for zone in NAMED_TIME_ZONES:

        distance = math.sqrt(
            (lat - zone[0]) ** 2 +
            (lon - zone[1]) ** 2
        )

        if distance < best_dist:
            best_dist = distance
            best_zone = zone

    if (
        best_zone is not None
        and best_dist <= best_zone[2]
    ):

        blend = 1 - (
            best_dist / best_zone[2]
        )

        ndvi = (
            best_zone[5] * blend
            + 0.35 * (1 - blend)
        )

        ndwi = (
            0.30 * blend
            + 0.15 * (1 - blend)
        )

        elevation = (
            best_zone[6] * blend
            + 250 * (1 - blend)
        )

        slope = (
            best_zone[7] * blend
            + 3 * (1 - blend)
        )

        dist_forest = (
            best_zone[8] * blend
            + 4 * (1 - blend)
        )

        dist_water = (
            best_zone[9] * blend
            + 3.5 * (1 - blend)
        )

        dist_road = (
            best_zone[10] * blend
            + 1.5 * (1 - blend)
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

    # No hardcoded flat fallback here — decay smoothly based on distance
    # to the nearest known zone (from either list), so even when the
    # live Overpass/elevation lookup fails, results still vary by
    # location instead of collapsing to one identical value for most
    # of India.
    best_dist = float("inf")

    for zone in FOREST_ZONES:
        d = math.sqrt(
            (lat - zone[0]) ** 2 +
            (lon - zone[1]) ** 2
        )
        best_dist = min(best_dist, d)

    for zone in NAMED_TIME_ZONES:
        d = math.sqrt(
            (lat - zone[0]) ** 2 +
            (lon - zone[1]) ** 2
        )
        best_dist = min(best_dist, d)

    dist_factor = min(1.0, best_dist / 3.0)  # decays over ~3 degrees (~330km)

    return (
        0.60 - dist_factor * 0.42,   # ndvi
        0.30 - dist_factor * 0.22,   # ndwi
        600 - dist_factor * 300,     # elevation
        12 - dist_factor * 10,       # slope
        1.5 + dist_factor * 7.5,     # dist_forest
        0.8 + dist_factor * 4.2,     # dist_water
        2.0 - dist_factor * 1.6,     # dist_road
    )


def build_features(
    lat,
    lon,
    ndvi,
    ndwi,
    elevation,
    slope,
    dist_forest,
    dist_water,
    dist_road
):

    vwr = ndvi / (
        ndwi + 0.01
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

    return [
        lat,
        lon,
        ndvi,
        ndwi,
        elevation,
        slope,
        dist_forest,
        dist_water,
        dist_road,
        vwr,
        tri,
        nld,
        has,
        esi,
        ndvi_ndwi_interaction,
        veg_water_risk,
        isolation_index,
        terrain_ratio,
        human_pressure,
        eco_stress,
        slope_elev_risk
    ]


def get_location_name(lat, lon):

    places = [
        (z[0], z[1], z[9])
        for z in FOREST_ZONES
    ]

    places += [
        (z[0], z[1], z[4])
        for z in NAMED_TIME_ZONES
    ]

    places += [
        (z[0], z[1], z[3])
        for z in URBAN_ZONES
    ]

    best_distance = float("inf")

    best_name = (
        f"({lat:.4f}, {lon:.4f})"
    )

    for plat, plon, name in places:

        distance = math.sqrt(
            (lat - plat) ** 2 +
            (lon - plon) ** 2
        )

        if distance < best_distance:

            best_distance = distance
            best_name = name

    if best_distance > 0.6:

        best_name = (
            f"({lat:.4f}, {lon:.4f})"
        )

    return best_name


def run_prediction(
    lat: float,
    lon: float,
    hour: Optional[int] = None,
    minute: Optional[int] = None
):

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

    if not 0 <= hour_used <= 23:
        raise ValueError(
            "Hour must be between 0 and 23"
        )

    if not 0 <= minute_used <= 59:
        raise ValueError(
            "Minute must be between 0 and 59"
        )

    urban, urban_name = is_urban(
        lat,
        lon
    )

    if urban:

        return {
            "risk": "LOW",
            "probability": 5.0,
            "base_probability": 5.0,
            "time_multiplier": 1.0,
            "time_profile": "urban",
            "hour_used": hour_used,
            "minute_used": minute_used,
            "time_used": (
                f"{hour_used:02d}:"
                f"{minute_used:02d}"
            ),
            "location": urban_name,
            "driver": "urban_area",
            "used_ml_model": False,
            "used_fallback_terrain": False,
            "lat": lat,
            "lon": lon
        }

    profile, multiplier = (
        get_exact_time_multiplier(
            lat,
            lon,
            hour_used,
            minute_used
        )
    )

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

    features = build_features(
        lat,
        lon,
        ndvi,
        ndwi,
        elevation,
        slope,
        dist_forest,
        dist_water,
        dist_road
    )

    dataframe = pd.DataFrame(
        [features],
        columns=FEATURES
    )

    scaled = scaler.transform(
        dataframe
    )

    base_probability = float(
        model.predict_proba(
            scaled
        )[0][1] * 100
    )

    adjusted_probability = (
        base_probability *
        multiplier
    )

    adjusted_probability = max(
        0,
        min(
            100,
            adjusted_probability
        )
    )

    if adjusted_probability >= 70:
        risk = "HIGH"
    elif adjusted_probability >= 40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    if ndvi > 0.65:
        driver = "vegetation"
    elif dist_forest < 2:
        driver = "forest_proximity"
    elif dist_water < 1:
        driver = "water_proximity"
    else:
        driver = "environmental_features"

    return {
        "risk": risk,
        "probability": round(
            adjusted_probability,
            2
        ),
        "base_probability": round(
            base_probability,
            2
        ),
        "time_multiplier": round(
            multiplier,
            4
        ),
        "time_profile": profile,
        "hour_used": hour_used,
        "minute_used": minute_used,
        "time_used": (
            f"{hour_used:02d}:"
            f"{minute_used:02d}"
        ),
        "location": get_location_name(
            lat,
            lon
        ),
        "driver": driver,
        "used_ml_model": True,
        "used_fallback_terrain": used_fallback,
        "lat": lat,
        "lon": lon
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/")
def root():
    return {
        "name": "HWC Prediction API",
        "status": "live"
    }


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


class EmailRequest(BaseModel):
    email: str


class VerifyRequest(BaseModel):
    email: str
    otp: str


class PendingRegistrationRequest(BaseModel):
    username: str
    email: str


def normalize_email(email: str) -> str:
    return email.strip().lower()


def validate_email(email: str) -> bool:
    return bool(
        re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            email
        )
    )


def send_email_otp(
    to_email: str,
    otp: str
):
    response = requests.post(
        "https://api.brevo.com/v3/smtp/email",

        headers={
            "api-key": BREVO_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json"
        },

        json={
            "sender": {
                "email": SENDER_EMAIL,
                "name": "HWC Alert"
            },

            "to": [
                {
                    "email": to_email
                }
            ],

            "subject":
                "HWC Alert - Your Login Code",

            "textContent":
                f"Your HWC Alert login code is: {otp}\n\n"
                "Expires in 5 minutes."
        },

        timeout=15
    )

    if response.status_code >= 300:
        raise Exception(
            f"Brevo API error "
            f"{response.status_code}: "
            f"{response.text}"
        )


def send_email_verification_link(
    to_email: str,
    username: str,
    verification_url: str
):
    html = f"""
    <!DOCTYPE html>
    <html>
      <body style="font-family:Arial,sans-serif;line-height:1.6;color:#222;">
        <h2>Welcome to WILDORA</h2>
        <p>Hello {username},</p>
        <p>
          Click the button below to verify your email address and finish
          creating your WILDORA account.
        </p>
        <p>
          <a href="{verification_url}"
             style="display:inline-block;padding:12px 20px;
                    background:#2e7d32;color:white;text-decoration:none;
                    border-radius:6px;">
            Verify Email &amp; Create Account
          </a>
        </p>
        <p>This link expires in 30 minutes and can be used only once.</p>
        <p>If you did not request this account, you can ignore this email.</p>
      </body>
    </html>
    """

    text = (
        f"Hello {username},\n\n"
        "Verify your email and finish creating your WILDORA account:\n"
        f"{verification_url}\n\n"
        "This link expires in 30 minutes and can be used only once.\n"
    )

    response = requests.post(
        "https://api.brevo.com/v3/smtp/email",

        headers={
            "api-key": BREVO_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json"
        },

        json={
            "sender": {
                "email": SENDER_EMAIL,
                "name": "HWC Alert"
            },
            "to": [
                {
                    "email": to_email
                }
            ],
            "subject": "WILDORA - Verify Your Email",
            "htmlContent": html,
            "textContent": text
        },

        timeout=15
    )

    if response.status_code >= 300:
        raise Exception(
            f"Brevo API error "
            f"{response.status_code}: "
            f"{response.text}"
        )


def generate_otp(
    email: str,
    time_step: int
):
    msg = (
        f"{email.lower().strip()}:"
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


@app.post("/register-pending")
def register_pending(
    req: PendingRegistrationRequest
):
    username = req.username.strip()
    email = normalize_email(req.email)

    if not username:
        return {"error": "Username is required"}

    if len(username) > 100:
        return {"error": "Username is too long"}

    if not validate_email(email):
        return {"error": "Please enter a valid email address"}

    # Do not create a Firebase account yet. The account is created only
    # after the verification link is clicked.
    try:
        existing_user = firebase_auth.get_user_by_email(email)

        if existing_user.email_verified:
            return {
                "error": "An account with this email already exists. Please sign in."
            }

        # A pre-existing unverified Firebase user is not expected with this
        # flow, but remove it so the verification process remains consistent.
        try:
            firebase_auth.delete_user(existing_user.uid)
        except Exception:
            pass

    except firebase_auth.UserNotFoundError:
        pass

    # Invalidate any older pending registrations for this email.
    pending_query = (
        db.collection("pending_registrations")
        .where("email", "==", email)
        .stream()
    )

    for document in pending_query:
        try:
            document.reference.delete()
        except Exception:
            pass

    token = secrets.token_urlsafe(48)
    expires_at = time.time() + VERIFICATION_TTL_SECONDS

    db.collection("pending_registrations").document(token).set({
        "username": username,
        "email": email,
        "created_at": firestore.SERVER_TIMESTAMP,
        "expires_at": expires_at,
        "used": False
    })

    verification_url = (
        f"{VERIFICATION_BASE_URL}/verify-email"
        f"?token={quote(token, safe='')}"
    )

    try:
        send_email_verification_link(
            email,
            username,
            verification_url
        )
    except Exception as error:
        try:
            db.collection("pending_registrations").document(token).delete()
        except Exception:
            pass

        return {
            "error": f"Failed to send verification email: {error}"
        }

    return {
        "message": "Verification email sent"
    }


@app.get("/verify-email", response_class=HTMLResponse)
def verify_email(token: str):
    token = token.strip()

    if not token:
        return HTMLResponse(
            content="""
            <html>
              <body style="font-family:Arial;text-align:center;padding:40px;">
                <h2>Verification failed</h2>
                <p>The verification link is missing a token.</p>
              </body>
            </html>
            """,
            status_code=400
        )

    document_ref = db.collection("pending_registrations").document(token)
    document = document_ref.get()

    if not document.exists:
        return HTMLResponse(
            content="""
            <html>
              <body style="font-family:Arial;text-align:center;padding:40px;">
                <h2>Link invalid or already used</h2>
                <p>Please return to WILDORA and create a new account if needed.</p>
              </body>
            </html>
            """,
            status_code=400
        )

    data = document.to_dict() or {}

    if data.get("used"):
        return HTMLResponse(
            content="""
            <html>
              <body style="font-family:Arial;text-align:center;padding:40px;">
                <h2>Link already used</h2>
                <p>Your email verification link has already been used.</p>
              </body>
            </html>
            """,
            status_code=400
        )

    expires_at = float(data.get("expires_at", 0))

    if time.time() > expires_at:
        try:
            document_ref.delete()
        except Exception:
            pass

        return HTMLResponse(
            content="""
            <html>
              <body style="font-family:Arial;text-align:center;padding:40px;">
                <h2>Link expired</h2>
                <p>This verification link has expired. Please create the account again in WILDORA.</p>
              </body>
            </html>
            """,
            status_code=400
        )

    email = normalize_email(data.get("email", ""))
    username = str(data.get("username", "")).strip()

    if not email or not username:
        return HTMLResponse(
            content="""
            <html>
              <body style="font-family:Arial;text-align:center;padding:40px;">
                <h2>Verification failed</h2>
                <p>The registration information is incomplete.</p>
              </body>
            </html>
            """,
            status_code=400
        )

    try:
        # If a verified account was created while this link was pending,
        # do not create a duplicate account.
        try:
            existing_user = firebase_auth.get_user_by_email(email)

            if existing_user.email_verified:
                document_ref.update({"used": True})
                return HTMLResponse(
                    content="""
                    <html>
                      <body style="font-family:Arial;text-align:center;padding:40px;">
                        <h2>Email already verified</h2>
                        <p>Your WILDORA account already exists.</p>
                        <p>You can now open the WILDORA app and sign in.</p>
                      </body>
                    </html>
                    """
                )

            firebase_auth.delete_user(existing_user.uid)

        except firebase_auth.UserNotFoundError:
            pass

        # This is the first point at which the Firebase account is created.
        user_record = firebase_auth.create_user(
            email=email,
            display_name=username,
            email_verified=True
        )

        # Keep a simple application profile in Firestore.
        db.collection("users").document(user_record.uid).set({
            "uid": user_record.uid,
            "username": username,
            "email": email,
            "email_verified": True,
            "created_at": firestore.SERVER_TIMESTAMP
        })

        document_ref.update({
            "used": True,
            "verified_uid": user_record.uid,
            "verified_at": firestore.SERVER_TIMESTAMP
        })

        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html>
              <body style="font-family:Arial,sans-serif;text-align:center;padding:40px;">
                <h2 style="color:#2e7d32;">Email Verified Successfully!</h2>
                <p>Welcome, {username}.</p>
                <p>Your WILDORA account has been created.</p>
                <p><strong>You can now open the WILDORA app and sign in with your email.</strong></p>
              </body>
            </html>
            """
        )

    except Exception as error:
        print("Email verification failed:", error)

        return HTMLResponse(
            content="""
            <html>
              <body style="font-family:Arial;text-align:center;padding:40px;">
                <h2>Verification failed</h2>
                <p>We could not complete account creation.</p>
                <p>Please try creating the account again in WILDORA.</p>
              </body>
            </html>
            """,
            status_code=500
        )


@app.post("/send-otp")
def send_otp(
    req: EmailRequest
):
    email = normalize_email(req.email)

    if not validate_email(email):
        return {"error": "Please enter a valid email address"}

    # Sign-in OTP is allowed only for accounts that have completed
    # email-link verification.
    try:
        user = firebase_auth.get_user_by_email(email)
    except firebase_auth.UserNotFoundError:
        return {
            "error": "No WILDORA account found for this email. Please create an account first."
        }

    if not user.email_verified:
        return {
            "error": "Please verify your email before signing in."
        }

    time_step = int(
        time.time()
        // OTP_STEP_SECONDS
    )

    otp = generate_otp(
        email,
        time_step
    )

    try:
        send_email_otp(
            email,
            otp
        )

    except Exception as error:
        return {
            "error":
                f"Failed to send email: {error}"
        }

    return {
        "message": "OTP sent"
    }


@app.post("/verify-otp")
def verify_otp(
    req: VerifyRequest
):
    email = normalize_email(req.email)
    entered = req.otp.strip()

    try:
        user = firebase_auth.get_user_by_email(email)
    except firebase_auth.UserNotFoundError:
        return {
            "error":
                "No WILDORA account found for this email."
        }

    if not user.email_verified:
        return {
            "error":
                "Please verify your email before signing in."
        }

    current_step = int(
        time.time()
        // OTP_STEP_SECONDS
    )

    valid_otp = (
        generate_otp(
            email,
            current_step
        ) == entered
        or
        generate_otp(
            email,
            current_step - 1
        ) == entered
    )

    if not valid_otp:
        return {
            "error":
                "Incorrect or expired OTP"
        }

    try:
        custom_token = firebase_auth.create_custom_token(
            user.uid
        )

        # create_custom_token returns bytes in some firebase-admin versions.
        if isinstance(custom_token, bytes):
            custom_token = custom_token.decode("utf-8")

        return {
            "message": "Verified",
            "uid": user.uid,
            "email": user.email,
            "username": user.display_name or "",
            "custom_token": custom_token
        }

    except Exception as error:
        print("Custom token creation failed:", error)

        return {
            "error":
                "Could not create sign-in token"
        }
