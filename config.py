"""Application configuration loaded from environment variables.

Uses python-dotenv to load a .env file (if present) and falls back to
sensible defaults for all settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root if it exists
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# --- ML Model Paths ---
MODEL_PATH: str = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "ddos_model.joblib"))
SCALER_PATH: str = os.getenv("SCALER_PATH", str(BASE_DIR / "models" / "scaler.joblib"))
METADATA_PATH: str = os.getenv("METADATA_PATH", str(BASE_DIR / "models" / "metadata.json"))

# --- Database ---
DB_PATH: str = os.getenv("DB_PATH", str(BASE_DIR / "data" / "attacks.db"))

# --- Threat Feed ---
FEED_INTERVAL_SEC: int = int(os.getenv("FEED_INTERVAL_SEC", "300"))
FIREHOL_URL: str = os.getenv(
    "FIREHOL_URL",
    "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_level1.netset",
)

# --- Geolocation ---
GEO_BATCH_SIZE: int = int(os.getenv("GEO_BATCH_SIZE", "50"))
