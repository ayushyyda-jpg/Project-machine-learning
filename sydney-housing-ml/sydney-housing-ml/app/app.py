"""
Sydney Housing Price Estimator: Flask web app for the SIT307 ML mini project.

Serves the Gradient Boosting model trained in housing_price_prediction.ipynb.
Two ways to use it:
  1. Fill in the form for one property -> predicted price, likely range,
     comparable sales from the training data, and caution flags.
  2. Upload a CSV of properties -> a table of predictions (downloadable).

Run from the project folder:   python app/app.py
Then open:                     http://127.0.0.1:5000
"""
from __future__ import annotations

import base64
import io
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, Response, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import housing_features as hf  # noqa: E402  (shared with the notebook)

MODEL = joblib.load(ROOT / "app" / "model" / "house_price_model.joblib")
META = json.loads((ROOT / "app" / "model" / "model_meta.json").read_text())
SALES = pd.read_csv(ROOT / "data" / "sydney_sold_clean.csv", parse_dates=["sold_date"])
SALES["dwelling_class"] = [hf.dwelling_class(t, a) for t, a in zip(SALES.property_type, SALES.address)]
Q_LOW, Q_HIGH = META["interval_log_quantiles"]

if sklearn.__version__ != META["sklearn_version"]:
    print(f"WARNING: model saved with scikit-learn {META['sklearn_version']}, "
          f"running {sklearn.__version__}. Re-run the notebook if predictions look wrong.")

PROPERTY_TYPES = ["House", "Semi-detached", "Duplex", "Townhouse", "Villa", "Apartment / Unit / Flat"]
SALE_METHODS = ["private treaty", "auction", "prior to auction"]
CSV_COLUMNS = ["suburb", "property_type", "beds", "baths", "parking", "area_m2", "sale_method", "sold_date"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB upload limit


# ---------------------------------------------------------------- helpers
def money(v: float) -> str:
    return f"${v:,.0f}"


def validate(df: pd.DataFrame) -> list[str]:
    """Hard input errors: these rows cannot be priced."""
    errors = []
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing column(s): {', '.join(missing)}"]
    for i, r in df.iterrows():
        row = f"Row {i + 1}: " if len(df) > 1 else ""
        if r.suburb not in hf.SUBURBS:
            errors.append(f"{row}suburb must be one of {', '.join(hf.SUBURBS)} (got '{r.suburb}').")
        if r.property_type not in PROPERTY_TYPES:
            errors.append(f"{row}property_type '{r.property_type}' is not recognised.")
        for col in ["beds", "baths", "parking"]:
            v = pd.to_numeric(r[col], errors="coerce")
            if pd.isna(v) or v < 0 or v > 20:
                errors.append(f"{row}{col} must be a number between 0 and 20.")
        if pd.isna(pd.to_datetime(r.sold_date, errors="coerce")):
            errors.append(f"{row}sold_date must be a date like 2026-10-01.")
    return errors


def caution_flags(raw: pd.Series, feats: pd.Series) -> list[str]:
    """Soft warnings: the model can price this, but it is outside its experience."""
    flags = []
    lo, hi = META["ranges"]["beds"]
    if not lo <= feats.beds <= hi:
        flags.append(f"{int(feats.beds)} bedrooms is outside the training range ({lo:.0f}–{hi:.0f}); "
                     "the model cannot extrapolate, so the estimate is likely too low or too high.")
    seg = f"{feats.suburb}_{feats.dwelling_class}"
    n = META["segment_counts"].get(seg, 0)
    if n < 10:
        flags.append(f"Only {n} {feats.dwelling_class.replace('_', '/')} sales in {feats.suburb} were in the training data.")
    if raw.property_type == "Semi-detached":
        flags.append("Semi-detached homes are grouped with houses; in testing the model over-valued a Bondi semi by 49%.")
    area = pd.to_numeric(raw.area_m2, errors="coerce")
    if feats.dwelling_class == "unit" and pd.notna(area):
        flags.append("Area is ignored for apartments (Domain areas mix floor area and whole-site land).")
    if pd.notna(feats.land_area_m2):
        alo, ahi = META["ranges"]["land_area_m2"]
        if not alo <= feats.land_area_m2 <= ahi:
            flags.append(f"Land area {feats.land_area_m2:.0f} m² is outside the training range ({alo:.0f}–{ahi:.0f} m²).")
        if feats.land_area_m2 >= 700:
            flags.append("Large blocks may carry development value the model cannot see (it under-valued a 739 m² "
                         "Parramatta development site by 41%).")
    end = pd.Timestamp(META["sale_dates"][1])
    when = pd.to_datetime(raw.sold_date)
    if when > end + pd.DateOffset(months=3):
        flags.append(f"The sale date is more than 3 months after the newest sale in the data ({end:%b %Y}); "
                     "market movement since then is not reflected.")
    return flags


def comparables(feats: pd.Series, k: int = 3) -> list[dict]:
    """Most similar real sales: same suburb and dwelling class, closest beds/baths, most recent."""
    pool = SALES[(SALES.suburb == feats.suburb) & (SALES.dwelling_class == feats.dwelling_class)].copy()
    if pool.empty:
        return []
    pool["distance"] = (pool.beds - feats.beds).abs() * 2 + (pool.baths - feats.baths).abs()
    pool = pool.sort_values(["distance", "sold_date"], ascending=[True, False]).head(k)
    return [{"address": f"{r.address}, {r.suburb}", "summary": f"{r.beds} bed · {r.baths} bath · {r.property_type}",
             "price": money(r.price), "date": f"{r.sold_date:%b %Y}", "url": r.url} for r in pool.itertuples()]


def predict_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "address" not in df.columns:
        df["address"] = ""
    df["address"] = df["address"].fillna("")
    feats = hf.engineer_features(df)
    pred = MODEL.predict(feats)
    out = df[CSV_COLUMNS].copy()
    out["predicted_price"] = pred.round(-3)
    out["likely_low"] = (pred * np.exp(Q_LOW)).round(-3)
    out["likely_high"] = (pred * np.exp(Q_HIGH)).round(-3)
    out["cautions"] = [" | ".join(caution_flags(r, f)) for (_, r), (_, f) in zip(df.iterrows(), feats.iterrows())]
    return out, feats


# ---------------------------------------------------------------- routes
@app.route("/", methods=["GET", "POST"])
def index():
    form = {"suburb": "Penrith", "property_type": "House", "beds": 3, "baths": 2, "parking": 1,
            "area_m2": "", "sale_method": "private treaty", "sold_month": date.today().strftime("%Y-%m")}
    result, errors = None, []
    if request.method == "POST":
        form.update({k: request.form.get(k, "").strip() for k in form})
        row = pd.DataFrame([{**form, "sold_date": f"{form['sold_month']}-15" if form["sold_month"] else ""}])
        row["area_m2"] = pd.to_numeric(row.area_m2, errors="coerce")
        errors = validate(row)
        if not errors:
            out, feats = predict_frame(row)
            r, f = out.iloc[0], feats.iloc[0]
            result = {
                "price": money(r.predicted_price), "low": money(r.likely_low), "high": money(r.likely_high),
                "flags": caution_flags(row.iloc[0], f), "comps": comparables(f),
                "segment": f"{f.dwelling_class.replace('_', '/')} in {f.suburb}",
                "marker": f"{100 * -Q_LOW / (Q_HIGH - Q_LOW):.0f}%",
            }
    return render_template("index.html", form=form, result=result, errors=errors, meta=META,
                           suburbs=hf.SUBURBS, types=PROPERTY_TYPES, methods=SALE_METHODS,
                           cv_mape=f"{META['cv_mape']:.0%}", tab="single")


@app.route("/batch", methods=["POST"])
def batch():
    errors, table, download = [], None, None
    file = request.files.get("file")
    if not file or not file.filename:
        errors = ["Choose a CSV file to upload."]
    else:
        try:
            df = pd.read_csv(file)
            df.columns = [c.strip().lower() for c in df.columns]
            errors = validate(df) if len(df) else ["The CSV has no rows."]
            if not errors:
                out, _ = predict_frame(df)
                download = base64.b64encode(out.to_csv(index=False).encode()).decode()
                show = out.copy()
                for c in ["predicted_price", "likely_low", "likely_high"]:
                    show[c] = show[c].map(money)
                table = show.to_dict("records")
        except Exception as exc:  # malformed CSV
            errors = [f"Could not read the CSV: {exc}"]
    return render_template("index.html", form=None, result=None, errors=errors, meta=META,
                           suburbs=hf.SUBURBS, types=PROPERTY_TYPES, methods=SALE_METHODS,
                           cv_mape=f"{META['cv_mape']:.0%}", tab="batch", table=table, download=download)


@app.route("/template.csv")
def template_csv():
    example = pd.DataFrame([
        {"suburb": "Bondi", "property_type": "Apartment / Unit / Flat", "beds": 2, "baths": 1, "parking": 1,
         "area_m2": "", "sale_method": "auction", "sold_date": "2026-10-15"},
        {"suburb": "Parramatta", "property_type": "House", "beds": 4, "baths": 2, "parking": 2,
         "area_m2": 550, "sale_method": "private treaty", "sold_date": "2026-10-15"},
        {"suburb": "Penrith", "property_type": "Townhouse", "beds": 3, "baths": 2, "parking": 2,
         "area_m2": "", "sale_method": "private treaty", "sold_date": "2026-10-15"},
    ])
    return Response(example.to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=properties_template.csv"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: POST one property object (or a list) with the CSV columns."""
    payload = request.get_json(force=True)
    df = pd.DataFrame(payload if isinstance(payload, list) else [payload])
    errors = validate(df)
    if errors:
        return jsonify({"errors": errors}), 400
    out, _ = predict_frame(df)
    return jsonify(json.loads(out.to_json(orient="records")))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
