"""Dataset loading and time-safe feature engineering for Sri Lankan daily rain.

The data are supplied by a third-party GitHub author. Users should verify original
provenance and reuse rights; raw data are NOT redistributed with this repository.
"""
# AI-assisted implementation: OpenAI ChatGPT (GPT-6), 2026-11-22. See AI_DISCLOSURE.md.

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/Lahirulakshan129/"
    "SriLanka-Weather-Dataset/main/SriLanka_Weather_2020_2025.csv"
)
BASE = Path(__file__).resolve().parent
RAW_PATH = BASE / "data" / "sri_lanka_weather_2020_2025.csv"
NUMERIC_FEATURES = [
    "temp", "humidity", "precip", "windspeed", "rain_3d_total",
    "rain_7d_total", "rainy_days_7d", "season_sin", "season_cos",
]
CATEGORICAL_FEATURES = ["district"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
THRESHOLD_MM = 1.0


def download_data(path: Path = RAW_PATH, url: str = DATA_URL, force: bool = False) -> Path:
    """Save the sourced CSV locally once. No fabricated dataset fallback."""
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        text = response.content
        if not text.startswith(b"date,temp,humidity,precip,windspeed,district"):
            raise ValueError("Unexpected CSV header from data source")
        path.write_bytes(text)
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(
            "Unable to retrieve the public Sri Lankan weather CSV. "
            "Download it manually from the DATA_URL in data_utils.py "
            f"and save to {path}. Original error: {exc}"
        ) from exc
    return path


def load_raw(path: Path | None = None) -> pd.DataFrame:
    source = Path(path) if path is not None else download_data()
    data = pd.read_csv(source, low_memory=False)
    required = {"date", "temp", "humidity", "precip", "windspeed", "district"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Create *next-day* target using only current/prior-day predictors.

    A row at date T contains conditions on T, past rolling rainfall, and
    target rain on T+1. Rows with gaps in T/T+1 are excluded. Splitting by
    target_date prevents any future outcomes from entering train/validation.
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["district"] = df["district"].astype("string").str.strip()
    for col in ["temp", "humidity", "precip", "windspeed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "district", "precip"])
    df = df[(df["precip"] >= 0) & (df["date"].dt.year.between(2020, 2025))]
    df = df.sort_values(["district", "date"]).drop_duplicates(["district", "date"], keep="last")
    groups = df.groupby("district", sort=False, observed=True)
    df["rain_3d_total"] = groups["precip"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["rain_7d_total"] = groups["precip"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    df["rainy_days_7d"] = groups["precip"].transform(
        lambda s: s.ge(THRESHOLD_MM).rolling(7, min_periods=1).sum()
    )
    df["target_date"] = groups["date"].shift(-1)
    df["next_precip"] = groups["precip"].shift(-1)
    df = df[(df["target_date"] - df["date"]).dt.days.eq(1)].copy()
    df["rain_tomorrow"] = (df["next_precip"] >= THRESHOLD_MM).astype(int)
    # Seasonal information for tomorrow is known beforehand and safe to use.
    day_of_year = df["target_date"].dt.dayofyear
    df["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    # For each district, a large fraction of missing readings would be unreliable.
    return df.sort_values(["target_date", "district"]).reset_index(drop=True)


def chronological_split(df: pd.DataFrame):
    years = df["target_date"].dt.year
    train = df[years <= 2023].copy()
    validation = df[years == 2024].copy()
    test = df[years == 2025].copy()
    for label, subset in [("training", train), ("validation", validation), ("test", test)]:
        if len(subset) < 50 or subset["rain_tomorrow"].nunique() < 2:
            raise ValueError(f"The {label} set needs at least 50 rows and both classes.")
    return train, validation, test
