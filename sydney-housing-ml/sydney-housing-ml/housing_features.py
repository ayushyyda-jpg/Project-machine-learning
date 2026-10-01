"""
Shared cleaning and feature-engineering code for the Sydney housing project.

Both the notebook (training) and the web app (prediction) import this file,
so a property is turned into model inputs in exactly the same way in both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUBURBS = ["Bondi", "Parramatta", "Penrith"]
DWELLING_CLASSES = ["house", "townhouse_villa", "unit"]

# Domain's raw "property type" labels -> three broad dwelling classes.
TYPE_TO_CLASS = {
    "House": "house",
    "Semi-detached": "house",
    "Duplex": "house",
    "Townhouse": "townhouse_villa",
    "Villa": "townhouse_villa",
    "Apartment / Unit / Flat": "unit",
}

# Reference date for the time-trend feature (first month in the dataset).
START_MONTH = pd.Timestamp("2025-09-01")

NUMERIC_FEATURES = ["beds", "baths", "parking", "land_area_m2",
                    "baths_per_bed", "months_since_start"]
BINARY_FEATURES = ["is_auction", "land_area_known"]
CATEGORICAL_FEATURES = ["suburb", "dwelling_class"]
MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def dwelling_class(property_type: str, address: str) -> str:
    """Map a Domain property type to a dwelling class.

    Domain sometimes labels strata apartments as 'House' (e.g. a top-floor
    apartment at 9/27 Castlefield St). A 'House' whose address carries a
    unit number ("9/27 ...") is therefore treated as a unit.
    """
    cls = TYPE_TO_CLASS.get(property_type, "unit")
    if cls == "house" and "/" in str(address):
        cls = "unit"
    return cls


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn cleaned listing rows into model-ready features.

    Expects columns: suburb, property_type, address, beds, baths, parking,
    area_m2, sale_method, sold_date. Works for one row (the app) or many.
    """
    out = pd.DataFrame(index=df.index)
    out["suburb"] = df["suburb"].astype(str)
    out["dwelling_class"] = [dwelling_class(t, a)
                             for t, a in zip(df["property_type"], df["address"])]

    out["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    out["baths"] = pd.to_numeric(df["baths"], errors="coerce")
    out["parking"] = pd.to_numeric(df["parking"], errors="coerce")
    out["baths_per_bed"] = out["baths"] / out["beds"].clip(lower=1)

    # Land area only means something for land-holding dwellings. Unit "areas"
    # on Domain are a mix of internal floor area and the whole strata site
    # (we saw 1,751 m2 and 2,364 m2 for 2-bed flats), so they are dropped.
    area = pd.to_numeric(df["area_m2"], errors="coerce")
    is_land = out["dwelling_class"].isin(["house", "townhouse_villa"])
    out["land_area_m2"] = area.where(is_land)
    out["land_area_known"] = out["land_area_m2"].notna().astype(int)

    out["is_auction"] = df["sale_method"].astype(str).str.contains(
        "auction", case=False).astype(int)
    sold = pd.to_datetime(df["sold_date"])
    out["months_since_start"] = ((sold.dt.year - START_MONTH.year) * 12
                                 + (sold.dt.month - START_MONTH.month)).astype(float)
    return out[MODEL_FEATURES]


def add_segment(X: pd.DataFrame) -> pd.DataFrame:
    """Add the suburb x dwelling-class interaction ("Bondi_house", ...).

    Used as the first step of every model pipeline rather than as a stored
    column, so permuting `suburb` or `dwelling_class` (permutation importance)
    also changes the interaction, and the app only has to supply raw inputs.
    """
    X = X.copy()
    X["segment"] = X["suburb"].astype(str) + "_" + X["dwelling_class"].astype(str)
    return X
