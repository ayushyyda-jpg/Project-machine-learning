# Sydney Housing Price Prediction and Decision Support System

SIT307 Machine Learning: ML Mini Project. Predicts the sale price of residential properties in **Bondi, Parramatta and Penrith** from 118 real sold listings collected from Domain.com.au (Sep 2025 – Sep 2026), and serves the model through a Flask web app.

## Project structure

```
├── housing_price_prediction.ipynb   # Parts 1–5: cleaning, EDA, features, models, errors, final model
├── housing_features.py              # Feature engineering shared by the notebook and the app
├── requirements.txt
├── data/
│   ├── raw_listings.csv, batch_*.txt   # Raw collection batches, as copied from Domain
│   ├── build_dataset.py                # Merges the batches -> sydney_sold_raw.csv
│   ├── sydney_sold_raw.csv             # 125 collected sales (uncleaned, with listing URLs)
│   └── sydney_sold_clean.csv           # 118 sales after cleaning (written by the notebook)
├── app/
│   ├── app.py                          # Flask application
│   ├── templates/index.html            # Page layout
│   ├── example_properties.csv          # Sample file for the CSV upload feature
│   └── model/                          # Saved model + metadata (written by the notebook)
├── figures/                            # Figures saved by the notebook
├── report/                             # PDF report and app screenshots
└── tools/                              # Notebook build/run helpers
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Rebuild the dataset (optional)

```bash
python data/build_dataset.py       # -> data/sydney_sold_raw.csv
```

## 2. Run the notebook

Open `housing_price_prediction.ipynb` in Jupyter and choose **Kernel → Restart & Run All**, or run it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace housing_price_prediction.ipynb
```

A full run takes 3–5 minutes (nested cross-validation). It writes the cleaned data, the figures, and the trained model to `app/model/`. All random steps use `random_state=42`.

## 3. Run the web app

```bash
python app/app.py
```

Open http://127.0.0.1:5000.

- **Single property:** pick suburb and property type, enter bedrooms, bathrooms, parking, optional land size, sale method and expected sale month, then press *Estimate price*. You get the estimate, a likely range, the three most similar real sales, and caution flags when the property is outside the model's experience.
- **Upload CSV:** download the template (or use `app/example_properties.csv`), upload it, and get a table of estimates that can be downloaded as CSV. Columns: `suburb, property_type, beds, baths, parking, area_m2, sale_method, sold_date`.
- **JSON API:** `POST /api/predict` with one property object (or a list) using the same fields.

The app loads the model saved by the notebook, so run the notebook at least once first (the repository already includes a trained model). Use the same scikit-learn version as in `requirements.txt`; the app prints a warning if the versions differ.

## Data source and limitations

Sale records were copied from public Domain.com.au sold-listing pages (price-disclosed sales only) and spot-checked against individual listings; each row keeps its listing URL. The data cover three suburbs over 13 months. The model must not be used for other suburbs or as a formal valuation. See Part 1 of the notebook for data quality, bias and limitations.
