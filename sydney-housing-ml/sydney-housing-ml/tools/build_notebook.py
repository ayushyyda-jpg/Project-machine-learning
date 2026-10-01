"""Builds housing_price_prediction.ipynb from the cell list below (no outputs).
Run tools/run_notebook.py afterwards to execute it and store outputs."""
import json, sys
from pathlib import Path

CELLS = []
def md(s): CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

RESULTS = json.loads(Path(__file__).with_name("results_text.json").read_text()) \
    if Path(__file__).with_name("results_text.json").exists() else {}
def md_result(key):
    md(RESULTS.get(key, f"*({key}: interpretation added after the first full run)*"))

# =========================================================================
md(r"""
# Sydney Housing Price Prediction and Decision Support System
**SIT307 Machine Learning: ML Mini Project (Distinction task)**
Author: Raghav · Data collected 30 Sep – 1 Oct 2026 from Domain.com.au

This notebook covers the full workflow: data collection and cleaning (Part 1), exploration and feature engineering (Part 2), model development with k-fold cross-validation (Part 3), analysis of the largest prediction failures (Part 4), and training and saving the model that the Flask web app serves (Part 5).

**How to run.** From the project folder: `pip install -r requirements.txt`, then open this notebook and choose *Restart & Run All* (or `jupyter nbconvert --to notebook --execute --inplace housing_price_prediction.ipynb`). Every random step uses `random_state=42`, so results are reproducible. A full run takes about 3–5 minutes on a laptop, mostly the nested cross-validation in Part 3.
""")

md(r"""
---
# Part 1 — Problem definition and data collection

## 1.1 The prediction problem
A real estate agency wants a quick, evidence-based estimate of what a property will sell for, to support appraisals, pricing discussions with vendors, and sanity checks on buyer offers. The task is **supervised regression**: given a property's characteristics (suburb, dwelling type, bedrooms, bathrooms, parking, land size, sale method and sale date), predict its **sale price in AUD**.

Because the agency works across very different markets, a useful model needs to be accurate in *relative* terms: a \$100k error matters far more on a \$500k Penrith unit than on a \$5m Bondi house. I therefore model **log(price)** and judge models mainly on **mean absolute percentage error (MAPE)**, alongside MAE and RMSE in dollars.

## 1.2 Suburb selection and why
I picked three suburbs where I would genuinely consider buying, each a very different Sydney market:

| Suburb | Character | Why it interests me as a buyer |
|---|---|---|
| **Bondi** (2026) | Eastern-suburbs beachside, ~7 km from the CBD. Old terraces and semis, small art-deco walk-up flats, very high land values. | Lifestyle and the beach. The question is what a modest budget actually buys there. |
| **Parramatta** (2150) | Sydney's second CBD, ~23 km west. High-rise apartment towers around the centre, older freestanding houses on large blocks in the surrounding streets. | Strong jobs hub with the new light rail and Metro West; apartments are relatively affordable. |
| **Penrith** (2750) | Outer-west regional centre, ~50 km from the CBD at the foot of the Blue Mountains. Detached houses on large blocks, newer apartment precincts (e.g. Lord Sheffield Circuit). | Most affordable entry point, and the Western Sydney Airport will change the area. |

**How suburb characteristics should shape prices.** Location is expected to dominate: proximity to the CBD and the coast pushes Bondi land values far above the west. Within a suburb, **dwelling type** matters most, because a house includes the land while a unit only shares it. **Bedrooms and bathrooms** proxy for size, **land area** for redevelopment potential (relevant in Parramatta and Penrith), and the **sale method** (auctions are common for sought-after stock) and **sale date** (market movement) add smaller effects.

## 1.3 How the data were collected
I used the public *Sold listings* pages on **Domain.com.au** for each suburb, sorted by sale date, with the "exclude price withheld" filter, and recorded each sale's price, sale date, sale method, bedrooms, bathrooms, parking, area (where shown), property type and listing URL. I also opened individual listing pages to verify a sample of records. Collection was done in batches (one per results page) and merged with `data/build_dataset.py`, which keeps the raw values untouched so every cleaning decision is visible below.
""")

code(r"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import inspect, json, platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import sklearn, joblib

import housing_features as hf

SEED = 42
np.random.seed(SEED)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
sns.set_theme(style="whitegrid", context="notebook")

SUBURB_ORDER = ["Bondi", "Parramatta", "Penrith"]
SUBURB_COLORS = {"Bondi": "#2a6f97", "Parramatta": "#e07a1f", "Penrith": "#3a8d4f"}
FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)

def money(v):
    return f"${v:,.0f}"

PRICE_TICKS = [3e5, 5e5, 1e6, 2e6, 4e6, 6e6]

def price_axis(ax, axis="y"):
    # Readable dollar ticks on a log-scaled price axis
    a = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    a.set_major_locator(mtick.FixedLocator(PRICE_TICKS))
    a.set_minor_locator(mtick.NullLocator())
    a.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1e6:g}m" if v >= 1e6 else f"${v/1e3:.0f}k"))

def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")

print("Python", platform.python_version(), "| pandas", pd.__version__,
      "| scikit-learn", sklearn.__version__, "| seaborn", sns.__version__)
""")

code(r"""
raw = pd.read_csv("data/sydney_sold_raw.csv", parse_dates=["sold_date"])
print(f"Raw records: {len(raw)} unique listing URLs")
print(raw.groupby("suburb").size().rename("records").to_string())
print(f"\nSale dates: {raw.sold_date.min():%d %b %Y} to {raw.sold_date.max():%d %b %Y}")
raw.head()
""")

md(r"""
## 1.4 Data quality audit
Before cleaning, I check what is missing, duplicated or suspicious.
""")

code(r"""
audit = pd.DataFrame({
    "missing": raw.isna().sum(),
    "missing_%": (raw.isna().mean() * 100).round(1),
})
display(audit[audit.missing > 0])

# Same sale listed twice under two listing IDs (agent re-listed the property)
dup = raw[raw.duplicated(subset=["address", "sold_date", "price"], keep=False)]
print("Same address, date and price listed twice:")
display(dup[["suburb", "address", "price", "sold_date", "sale_method", "url"]])

print("Domain property-type labels:")
print(raw.property_type.value_counts().to_string())
""")

code(r"""
# Retirement-village sales: 37 Mulgoa Rd and 90 Lethbridge St, Penrith are retirement villages.
retire_mask = (raw.property_type.eq("Retirement Living")
               | raw.address.str.contains("37 Mulgoa Road|90 Lethbridge Street"))
print("Retirement-village sales:")
display(raw.loc[retire_mask, ["address", "price", "property_type", "beds", "area_m2"]])

# 'House' labels with a unit number in the address
mislabel = raw[raw.property_type.eq("House") & raw.address.str.contains("/")]
print("Labelled 'House' by Domain but with a unit number:")
display(mislabel[["suburb", "address", "price", "beds", "property_type"]])

# Implausible unit areas (whole strata site instead of the apartment)
units = raw.property_type.eq("Apartment / Unit / Flat")
print("Apartment 'areas' recorded on Domain (m²):")
print(raw.loc[units & raw.area_m2.notna(), ["address", "beds", "area_m2"]].to_string(index=False))
""")

md(r"""
### Quality issues, challenges and sources of bias

**Missing information.** Area is missing for about 60% of records. Domain shows land area only when the agent enters it, and for apartments the field mixes internal floor area with the size of the whole strata site (two 2-bed flats show 1,751 m² and 2,364 m², clearly the building's land). Parking was absent for 3 Bondi houses.

**Errors in the source.** Domain labels some strata apartments as "House": 9/27 Castlefield St is described by its agent as a top-floor apartment. One Bondi sale (4/11 Flood St) appears twice under two listing IDs. Four retirement-village units (37 Mulgoa Rd, 90 Lethbridge St) sell under leasehold or licence arrangements, so their prices are not comparable freehold market values.

**Challenges in collection.** About half of all sales on Domain have *price withheld*, so I filtered to disclosed prices. Search pages sometimes disagreed with the listing page (344 Birrell St showed "Price Withheld" in the search grid but $5.86m on its own page), so I spot-checked records against individual listing pages; all four checks matched exactly. Results pages are capped, so to get enough houses I also used the "house" and "apartment" filtered pages.

**Potential bias and limits on validity.**
- *Selection bias from withheld prices.* Vendors withhold prices more often for disappointing results or prestige sales, so the dataset may under-represent both tails.
- *Sampling design.* I deliberately pulled house pages to balance the mix, so the house/unit ratio does not reflect each suburb's true sales mix. Suburb medians computed here are not market medians.
- *Different time windows.* Bondi and Parramatta houses sell less often, so their records go back further (to Sep 2025) than Penrith units (Jun–Sep 2026). Time and suburb are partly confounded.
- *Small sample.* ~118 usable sales means every estimate has wide uncertainty, and rare property types (townhouses, villas, dual-occupancy) have only a handful of examples.
- *Missing value drivers.* Condition, renovation, views, floor level, aspect, street quality and strata levies are not captured at all.
""")

code(r"""
log = [("Raw records (unique URLs)", len(raw))]
df = raw.copy()

df = df.drop_duplicates(subset=["address", "sold_date", "price"], keep="first")
log.append(("Drop duplicate listing of the same sale", len(df)))

df = df[~(df.property_type.eq("Retirement Living")
          | df.address.str.contains("37 Mulgoa Road|90 Lethbridge Street"))]
log.append(("Drop retirement-village (leasehold) sales", len(df)))

df = df.reset_index(drop=True)
cleaning_log = pd.DataFrame(log, columns=["step", "rows remaining"])
display(cleaning_log)

df.to_csv("data/sydney_sold_clean.csv", index=False)
counts = df.suburb.value_counts().reindex(SUBURB_ORDER)
print(counts.to_string())
assert len(df) >= 100 and counts.min() >= 30, "Brief requires >=100 sales and >=30 per suburb"
print(f"\nRequirement met: {len(df)} sales, minimum {counts.min()} per suburb.")
""")

md(r"""
Mislabelled property types and implausible unit areas are **not** dropped; they are corrected during feature engineering (Part 2), because the sale itself is valid. Outliers are examined in Part 2 and kept unless they are data errors.

---
# Part 2 — Data understanding and feature engineering
""")

code(r"""
df["dwelling_class"] = [hf.dwelling_class(t, a) for t, a in zip(df.property_type, df.address)]

summary = (df.groupby(["suburb", "dwelling_class"]).price
             .agg(n="count", median="median", mean="mean", min="min", max="max")
             .reindex(SUBURB_ORDER, level=0))
summary.style.format({c: "${:,.0f}" for c in ["median", "mean", "min", "max"]})
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
sns.histplot(df.price / 1e6, bins=30, ax=axes[0], color="#555")
axes[0].set(title="Sale price is strongly right-skewed", xlabel="Sale price ($m)", ylabel="Number of sales")
sns.histplot(data=df, x=np.log10(df.price), hue="suburb", hue_order=SUBURB_ORDER,
             palette=SUBURB_COLORS, bins=25, multiple="stack", ax=axes[1])
ticks = [3e5, 5e5, 1e6, 2e6, 4e6, 6e6]
axes[1].set_xticks(np.log10(ticks), ["$300k", "$500k", "$1m", "$2m", "$4m", "$6m"])
axes[1].set(title="On a log scale the suburbs separate into distinct markets", xlabel="Sale price (log scale)", ylabel="Number of sales")
plt.tight_layout(); save(fig, "fig1_price_distribution"); plt.show()

print(f"Skewness of price: {df.price.skew():.2f}  |  skewness of log(price): {np.log(df.price).skew():.2f}")
""")

md(r"""
**Distribution.** Prices run from about \$320k to \$6.3m with a long right tail (skewness above 1.5). On a log scale the distribution is much more symmetric and clearly multi-modal, roughly one cluster per suburb and dwelling type. This motivates modelling **log(price)**: errors become proportional, the Bondi houses no longer dominate the loss, and linear relationships become plausible.
""")

code(r"""
fig, ax = plt.subplots(figsize=(11, 4.8))
order = ["unit", "townhouse_villa", "house"]
sns.boxplot(data=df, x="suburb", y="price", hue="dwelling_class", order=SUBURB_ORDER,
            hue_order=order, ax=ax, showfliers=False, palette="Greys")
sns.stripplot(data=df, x="suburb", y="price", hue="dwelling_class", order=SUBURB_ORDER,
              hue_order=order, dodge=True, ax=ax, palette={c: "#c0392b" for c in order}, size=4, alpha=.7, legend=False)
price_axis(ax)
ax.set(title="Price by suburb and dwelling type (log scale)", xlabel="", ylabel="Sale price")
ax.legend(title="Dwelling class", loc="upper right")
plt.tight_layout(); save(fig, "fig2_suburb_type"); plt.show()

ratio = df.groupby(["suburb", "dwelling_class"]).price.median().unstack()
print("Median house price / median unit price:")
print((ratio["house"] / ratio["unit"]).reindex(SUBURB_ORDER).round(2).to_string())
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, sub in zip(axes, SUBURB_ORDER):
    d = df[df.suburb == sub]
    sns.stripplot(data=d, x="beds", y="price", hue="dwelling_class", hue_order=order,
                  ax=ax, jitter=.15, size=6, palette={"unit": "#999", "townhouse_villa": "#e0a91f", "house": SUBURB_COLORS[sub]})
    price_axis(ax); ax.set_title(sub); ax.set_xlabel("Bedrooms")
    if ax is not axes[0]: ax.get_legend().remove()
axes[0].set_ylabel("Sale price (log scale)")
fig.suptitle("More bedrooms means a higher price, but the step size depends on suburb and type", y=1.02)
plt.tight_layout(); save(fig, "fig3_beds"); plt.show()

print("Spearman correlation with price, within each suburb:")
corr = pd.DataFrame({s: df[df.suburb == s][["beds", "baths", "parking"]].corrwith(df[df.suburb == s].price, method="spearman")
                     for s in SUBURB_ORDER}).T
print(corr.round(2).to_string())
""")

code(r"""
df["month"] = df.sold_date.dt.to_period("M").dt.to_timestamp()
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
for sub in SUBURB_ORDER:
    d = df[df.suburb == sub]
    axes[0].scatter(d.sold_date, d.price, s=18, alpha=.6, color=SUBURB_COLORS[sub], label=sub)
price_axis(axes[0])
axes[0].set(title="Sales over time", ylabel="Sale price (log scale)", xlabel="Sale date")
axes[0].legend()
counts_m = df.groupby(["month", "suburb"]).size().unstack(fill_value=0).reindex(columns=SUBURB_ORDER)
counts_m.index = counts_m.index.strftime("%b %y")
counts_m.plot(kind="bar", stacked=True, ax=axes[1], color=[SUBURB_COLORS[s] for s in SUBURB_ORDER], width=.85)
axes[1].set(title="Records per month", xlabel="", ylabel="Sales")
plt.tight_layout(); save(fig, "fig4_time"); plt.show()

# Within-segment time trend: regress log price on months since start, inside each suburb x class cell
tmp = df.assign(t=hf.engineer_features(df).months_since_start, lp=np.log(df.price))
tmp["resid"] = tmp.lp - tmp.groupby(["suburb", "dwelling_class"]).lp.transform("mean")
slope = np.polyfit(tmp.t, tmp.resid, 1)[0]
print(f"Within-segment trend: {100*(np.exp(slope)-1):+.2f}% per month "
      f"(about {100*(np.exp(12*slope)-1):+.1f}% per year)")
""")

md_result("time_trend")

code(r"""
# Outliers: robust z-score of log price within each suburb x dwelling-class segment
lp = np.log(df.price)
grp = df.groupby(["suburb", "dwelling_class"])
med = grp.price.transform(lambda s: np.log(s).median())
mad = grp.price.transform(lambda s: (np.log(s) - np.log(s).median()).abs().median())
df["robust_z"] = (lp - med) / (1.4826 * mad.replace(0, np.nan))
outliers = df[df.robust_z.abs() > 2.5].sort_values("robust_z")
display(outliers[["suburb", "address", "dwelling_class", "beds", "baths", "area_m2", "price", "robust_z"]])
""")

md_result("outliers")

md(r"""
## 2.2 Hypothesis: the three most influential variables (stated before feature engineering)

1. **Suburb (location).** Land value is the biggest component of Sydney property prices, and land value is set by location: CBD access, coast, schools, amenity. Figure 2 shows medians differing several-fold between suburbs for the same dwelling type.
2. **Dwelling type (house vs unit).** A house includes its own land; a unit shares a strata lot. In land-scarce Bondi this should create a huge gap, and a smaller one in Penrith.
3. **Number of bedrooms.** The best available proxy for internal size, which we do not have directly. Within each suburb, price rises steadily with bedrooms (Figure 3).

I expect bathrooms and parking to matter, but mostly as correlated size signals (they overlap with bedrooms), and land area to matter for houses but to be limited by missing values.

## 2.3 Feature engineering
The engineered features (code below, shared with the web app via `housing_features.py`):

| Feature | Reason |
|---|---|
| `dwelling_class` (house / townhouse_villa / unit) | Collapses Domain's 6 labels into classes with enough examples, and **corrects mislabels**: a "House" with a unit number becomes a unit. |
| `segment` = suburb × dwelling class | The house premium differs hugely by suburb (Figure 2), which a purely additive linear model cannot represent otherwise. Built inside the pipeline. |
| `land_area_m2` + `land_area_known` | Land size only for houses/townhouses; unit "areas" set to missing because they mix floor area and site area. The flag lets models learn whether missingness itself carries signal. Missing values are imputed with the median *inside* each CV fold to avoid leakage. |
| `baths_per_bed` | Separates well-appointed homes (ensuites) from older stock with the same bed count. |
| `is_auction` | Auctions (and sales "prior to auction") are used for in-demand properties. |
| `months_since_start` | Captures market movement over the 13-month window. |

I did **not** add "distance to CBD": with only three suburbs it is a perfect relabelling of the suburb indicator and adds no information. It would become valuable with many suburbs.
""")

code(r"""
print(inspect.getsource(hf.engineer_features))
X = hf.engineer_features(df)
y = df.price.values
print("Feature matrix:", X.shape)
X.head()
""")

code(r"""
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate, cross_val_predict, GridSearchCV, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

def preprocessor(scale_numeric):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return Pipeline([
        ("segment", FunctionTransformer(hf.add_segment)),
        ("columns", ColumnTransformer([
            ("num", Pipeline(num_steps), hf.NUMERIC_FEATURES),
            ("bin", "passthrough", hf.BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             hf.CATEGORICAL_FEATURES + ["segment"]),
        ], verbose_feature_names_out=False)),
    ])

def log_target(pipe):
    # Fit on log(price), predict back in dollars, so every metric is in $ terms
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)

SCORING = {"MAPE": "neg_mean_absolute_percentage_error", "MAE": "neg_mean_absolute_error",
           "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
""")

code(r"""
# Does feature engineering help? Same Ridge model, raw vs engineered inputs, 5-fold CV x 3 repeats.
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)

raw_cols = ["suburb", "property_type", "beds", "baths", "parking"]
X_rawfeat = df[raw_cols].copy()
X_rawfeat["parking"] = X_rawfeat.parking.fillna(X_rawfeat.parking.median())
raw_ridge = log_target(Pipeline([
    ("columns", ColumnTransformer([
        ("num", StandardScaler(), ["beds", "baths", "parking"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["suburb", "property_type"])])),
    ("model", Ridge(alpha=1.0))]))
eng_ridge = log_target(Pipeline([("prep", preprocessor(True)), ("model", Ridge(alpha=1.0))]))

rows = []
for name, model, data in [("Ridge, raw Domain fields", raw_ridge, X_rawfeat),
                          ("Ridge, engineered features", eng_ridge, X)]:
    r = cross_validate(model, data, y, cv=cv, scoring=SCORING)
    rows.append({"inputs": name, "MAPE": -r["test_MAPE"].mean(), "MAE": -r["test_MAE"].mean(), "R2": r["test_R2"].mean()})
fe_compare = pd.DataFrame(rows).set_index("inputs")
fe_compare.style.format({"MAPE": "{:.1%}", "MAE": "${:,.0f}", "R2": "{:.3f}"})
""")

md_result("fe_compare")

md(r"""
---
# Part 3 — Model development and evaluation

## 3.1 Model choice and expectations (written before training)

| Model | Approach | Strengths | Weaknesses for this data |
|---|---|---|---|
| **Ridge regression** on log(price) | Linear, parametric, L2-regularised | Few parameters, so stable on ~118 rows; interpretable (coefficients ≈ % price effects); the log transform makes the multiplicative structure (suburb × size) close to linear | Cannot learn non-linearities or interactions unless engineered (hence `segment`); extrapolates linearly |
| **Random Forest** | Bagging: average of many deep, decorrelated trees | Captures interactions and thresholds automatically; robust to outliers and scaling; averaging lowers variance | Needs data to split on; with ~15–25 rows per segment, leaves get tiny and it predicts "steps"; cannot extrapolate beyond the training range |
| **Gradient Boosting** | Boosting: shallow trees fitted sequentially to residuals | Often the most accurate on tabular data; shallow trees plus a low learning rate give fine control of complexity | Most hyperparameters, so the easiest to overfit a small sample; less stable from fold to fold |

These are three genuinely different ways of controlling the bias–variance trade-off: a global linear form (high bias, low variance), variance reduction by averaging (bagging), and bias reduction by sequential correction (boosting).

**Prediction.** With only ~118 sales and strong, mostly multiplicative structure, I expect **Ridge to be very competitive** and the **Random Forest to be slightly best**, because it can learn the suburb × type interactions without overfitting as easily as boosting. I expect **Gradient Boosting to show the largest train–validation gap**.

## 3.2 Evaluation design
- **Target:** log(price), back-transformed, so all metrics are reported in dollars.
- **Outer loop:** 5-fold CV repeated 3 times (15 fits per model) for stable estimates on a small dataset.
- **Inner loop (nested CV):** each training fold runs its own 5-fold `GridSearchCV` to choose hyperparameters, so the reported scores are not optimistically biased by tuning on the same data.
- **Metrics:** MAPE (primary, fair across markets), MAE and RMSE in dollars (RMSE punishes large misses), and R². A naive **segment-median baseline** (predict the median price of the same suburb × dwelling class in the training fold) shows how much the models add over a rule of thumb.
""")

code(r"""
from sklearn.base import BaseEstimator, RegressorMixin

class SegmentMedian(BaseEstimator, RegressorMixin):
    # Baseline: median training price of the same suburb x dwelling class
    def fit(self, X, y):
        seg = X.suburb + "_" + X.dwelling_class
        self.medians_ = pd.Series(y, index=X.index).groupby(seg).median()
        self.global_ = float(np.median(y))
        return self
    def predict(self, X):
        seg = X.suburb + "_" + X.dwelling_class
        return seg.map(self.medians_).fillna(self.global_).values

inner = KFold(n_splits=5, shuffle=True, random_state=SEED)
models = {
    "Ridge": GridSearchCV(
        log_target(Pipeline([("prep", preprocessor(True)), ("model", Ridge())])),
        {"regressor__model__alpha": [0.1, 0.3, 1, 3, 10, 30]},
        cv=inner, scoring="neg_mean_absolute_percentage_error"),
    "Random Forest": GridSearchCV(
        log_target(Pipeline([("prep", preprocessor(False)),
                             ("model", RandomForestRegressor(n_estimators=300, random_state=SEED))])),
        {"regressor__model__min_samples_leaf": [1, 3, 5],
         "regressor__model__max_features": [0.33, 0.66, 1.0]},
        cv=inner, scoring="neg_mean_absolute_percentage_error", n_jobs=-1),
    "Gradient Boosting": GridSearchCV(
        log_target(Pipeline([("prep", preprocessor(False)),
                             ("model", GradientBoostingRegressor(learning_rate=0.05, subsample=0.8, random_state=SEED))])),
        {"regressor__model__n_estimators": [100, 300],
         "regressor__model__max_depth": [2, 3]},
        cv=inner, scoring="neg_mean_absolute_percentage_error", n_jobs=-1),
}

outer = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
cv_results = {"Baseline: segment median": cross_validate(SegmentMedian(), X, y, cv=outer, scoring=SCORING, return_train_score=True)}
for name, model in models.items():
    cv_results[name] = cross_validate(model, X, y, cv=outer, scoring=SCORING,
                                      return_train_score=True, return_estimator=True)
    print(f"{name}: done")

def summarise(res):
    out = {}
    for m in SCORING:
        sign = 1 if m == "R2" else -1
        out[f"val {m}"] = sign * res[f"test_{m}"].mean()
        out[f"val {m} sd"] = res[f"test_{m}"].std()
        out[f"train {m}"] = sign * res[f"train_{m}"].mean()
    return out

results = pd.DataFrame({k: summarise(v) for k, v in cv_results.items()}).T
table = results[["train MAPE", "val MAPE", "val MAPE sd", "val MAE", "val RMSE", "train R2", "val R2"]]
table.style.format({"train MAPE": "{:.1%}", "val MAPE": "{:.1%}", "val MAPE sd": "±{:.1%}",
                    "val MAE": "${:,.0f}", "val RMSE": "${:,.0f}", "train R2": "{:.3f}", "val R2": "{:.3f}"})
""")

code(r"""
print("Hyperparameters chosen by the inner CV in each of the 15 outer folds:")
for name in models:
    chosen = pd.Series([json.dumps({k.split("__")[-1]: v for k, v in est.best_params_.items()})
                        for est in cv_results[name]["estimator"]]).value_counts()
    print(f"\n{name}:\n" + chosen.to_string())
""")

code(r"""
# Paired comparison: all models were scored on the SAME 15 outer splits, so compare them fold by fold
fold_mape = pd.DataFrame({n: -cv_results[n]["test_MAPE"] for n in cv_results})
rows = []
for a, b in [("Gradient Boosting", "Ridge"), ("Gradient Boosting", "Random Forest"), ("Random Forest", "Ridge"), ("Ridge", "Baseline: segment median")]:
    d = fold_mape[a] - fold_mape[b]
    rows.append({"comparison": f"{a} vs {b}", "mean MAPE difference": d.mean(), "sd of difference": d.std(),
                 "folds where first is better": f"{(d < 0).sum()} / {len(d)}"})
paired = pd.DataFrame(rows).set_index("comparison")
paired.style.format({"mean MAPE difference": "{:+.2%}", "sd of difference": "{:.2%}"})
""")

code(r"""
names = list(cv_results)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.3))
x = np.arange(len(names))
tr = [results.loc[n, "train MAPE"] for n in names]
va = [results.loc[n, "val MAPE"] for n in names]
sd = [results.loc[n, "val MAPE sd"] for n in names]
axes[0].bar(x - .2, tr, .4, label="training folds", color="#bbb")
axes[0].bar(x + .2, va, .4, yerr=sd, capsize=4, label="validation folds", color="#2a6f97")
axes[0].set_xticks(x, [n.replace("Baseline: ", "Baseline:\n") for n in names])
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
axes[0].set(title="Train vs validation MAPE (lower is better)", ylabel="Mean absolute % error")
axes[0].legend()

box = pd.DataFrame({n: -cv_results[n]["test_MAPE"] for n in names})
sns.boxplot(data=box, ax=axes[1], color="#9ecae1")
axes[1].set_xticks(range(len(names)), [n.replace("Baseline: ", "Baseline:\n") for n in names])
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
axes[1].set(title="Spread of validation MAPE across the 15 outer folds", ylabel="")
plt.tight_layout(); save(fig, "fig5_cv_results"); plt.show()
""")

md(r"""
### Model complexity: under- and over-fitting
Validation curves vary one complexity parameter per model (5-fold CV, other settings fixed) and show training vs validation MAPE.
""")

code(r"""
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
curves = [
    ("Ridge: regularisation strength α", log_target(Pipeline([("prep", preprocessor(True)), ("model", Ridge())])),
     "regressor__model__alpha", [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000], True, "α (log scale, larger = simpler)"),
    ("Random Forest: minimum samples per leaf", log_target(Pipeline([("prep", preprocessor(False)), ("model", RandomForestRegressor(n_estimators=200, max_features=0.66, random_state=SEED))])),
     "regressor__model__min_samples_leaf", [1, 2, 3, 5, 8, 12, 20], False, "min_samples_leaf (larger = simpler)"),
    ("Gradient Boosting: number of trees (depth 3)", log_target(Pipeline([("prep", preprocessor(False)), ("model", GradientBoostingRegressor(learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED))])),
     "regressor__model__n_estimators", [10, 25, 50, 100, 200, 400, 800], False, "n_estimators (larger = more complex)"),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
curve_data = {}
for ax, (title, est, param, grid, logx, xlabel) in zip(axes, curves):
    tr, va = validation_curve(est, X, y, param_name=param, param_range=grid, cv=kf,
                              scoring="neg_mean_absolute_percentage_error", n_jobs=-1)
    tr, va = -tr.mean(1), -va.mean(1)
    curve_data[title] = pd.DataFrame({"train": tr, "validation": va}, index=grid)
    ax.plot(grid, tr, "o-", color="#999", label="train")
    ax.plot(grid, va, "o-", color="#2a6f97", label="validation")
    if logx: ax.set_xscale("log")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set(title=title, xlabel=xlabel)
axes[0].set_ylabel("MAPE"); axes[0].legend()
plt.tight_layout(); save(fig, "fig6_complexity"); plt.show()
for t, d in curve_data.items():
    print(t); print((d * 100).round(1).T.to_string()); print()
""")

md_result("cv_interpretation")

code(r"""
# Out-of-fold predictions (5-fold, each fold tuned by its own inner CV) for per-suburb errors and Part 4
oof_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {name: cross_val_predict(model, X, y, cv=oof_cv) for name, model in models.items()}
oof["Baseline: segment median"] = cross_val_predict(SegmentMedian(), X, y, cv=oof_cv)

by_suburb = pd.DataFrame({
    name: pd.Series(np.abs(pred - y) / y, index=df.index).groupby([df.suburb, df.dwelling_class]).mean()
    for name, pred in oof.items()})
by_suburb.insert(0, "n", df.groupby(["suburb", "dwelling_class"]).size())
by_suburb.style.format({c: "{:.1%}" for c in by_suburb.columns if c != "n"})
""")

code(r"""
# Which inputs matter? Permutation importance measured on held-out folds of the Ridge and Gradient Boosting models.
from sklearn.inspection import permutation_importance

def heldout_importance(estimator, n_repeats=10):
    imps = []
    for k, (tr_idx, te_idx) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(X)):
        est = sklearn.base.clone(estimator).fit(X.iloc[tr_idx], y[tr_idx])
        r = permutation_importance(est, X.iloc[te_idx], y[te_idx], n_repeats=n_repeats,
                                   scoring="neg_mean_absolute_percentage_error", random_state=SEED)
        imps.append(pd.Series(r.importances_mean, index=X.columns))
    return pd.concat(imps, axis=1).mean(axis=1)

best_ridge = log_target(Pipeline([("prep", preprocessor(True)), ("model", Ridge(alpha=1))]))
best_gb = log_target(Pipeline([("prep", preprocessor(False)), ("model", GradientBoostingRegressor(
    n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED))]))
imp = pd.DataFrame({"Ridge": heldout_importance(best_ridge), "Gradient Boosting": heldout_importance(best_gb)})
imp = imp.sort_values("Gradient Boosting")

fig, ax = plt.subplots(figsize=(9, 4.5))
imp.plot.barh(ax=ax, color=["#9ecae1", "#2a6f97"])
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set(title="Permutation importance on held-out folds\n(increase in MAPE when a feature is shuffled)", xlabel="Increase in MAPE")
plt.tight_layout(); save(fig, "fig7_importance"); plt.show()
(imp.sort_values("Gradient Boosting", ascending=False) * 100).round(2).rename(columns=lambda c: c + " (MAPE +pp)")
""")

md_result("importance")

md_result("expectations_revisited")

md(r"""
---
# Part 4 — Investigating prediction failures
The five largest **absolute** errors from the recommended model's out-of-fold predictions (each property predicted by a model that never saw it).
""")

code(r"""
best_name = RESULTS_BEST = "BEST_MODEL_PLACEHOLDER"
err = df[["suburb", "address", "dwelling_class", "property_type", "beds", "baths", "parking", "area_m2", "sale_method", "sold_date", "price", "url"]].copy()
err["predicted"] = oof[best_name]
err["error"] = err.predicted - err.price
err["abs_error"] = err.error.abs()
err["pct_error"] = err.error / err.price
seg_stats = df.groupby(["suburb", "dwelling_class"]).price.agg(seg_median="median", seg_n="count")
err = err.join(seg_stats, on=["suburb", "dwelling_class"])

top5 = err.nlargest(5, "abs_error")
show = top5[["suburb", "address", "property_type", "beds", "baths", "parking", "area_m2", "price", "predicted", "error", "pct_error", "seg_median", "seg_n"]]
display(show.style.format({"price": "${:,.0f}", "predicted": "${:,.0f}", "error": "${:+,.0f}",
                           "pct_error": "{:+.1%}", "seg_median": "${:,.0f}", "area_m2": "{:.0f}"}))

print("For comparison, the five largest percentage errors:")
display(err.reindex(err.pct_error.abs().nlargest(5).index)[["suburb", "address", "property_type", "beds", "price", "predicted", "pct_error"]]
        .style.format({"price": "${:,.0f}", "predicted": "${:,.0f}", "pct_error": "{:+.1%}"}))
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for sub in SUBURB_ORDER:
    d = err[err.suburb == sub]
    ax.scatter(d.price, d.predicted, s=22, alpha=.7, color=SUBURB_COLORS[sub], label=sub)
lims = [2.5e5, 8e6]
ax.plot(lims, lims, "k--", lw=1)
ax.fill_between(lims, [l * .8 for l in lims], [l * 1.2 for l in lims], color="grey", alpha=.12, label="±20% band")
for i, (_, r) in enumerate(top5.iterrows(), 1):
    ax.annotate(str(i), (r.price, r.predicted), xytext=(6, -10), textcoords="offset points", fontsize=11, weight="bold")
    ax.scatter(r.price, r.predicted, s=110, facecolors="none", edgecolors="crimson", lw=1.6)
price_axis(ax, "x"); price_axis(ax, "y")
ax.set(xlim=lims, ylim=lims, xlabel="Actual sale price", ylabel="Out-of-fold prediction",
       title=f"{best_name}: predicted vs actual (top-5 errors circled)")
ax.legend(loc="upper left")

seg_err = err.assign(segment=err.suburb + "\n" + err.dwelling_class)
sns.boxplot(data=seg_err, x="segment", y="pct_error", ax=axes[1], color="#9ecae1",
            order=sorted(seg_err.segment.unique()))
axes[1].axhline(0, color="k", lw=1)
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
axes[1].set(title="Percentage error by segment (positive = over-predicted)", xlabel="", ylabel="Prediction error")
axes[1].tick_params(axis="x", labelsize=8)
plt.tight_layout(); save(fig, "fig8_errors"); plt.show()

print("Share of properties predicted within ±10% / ±20%:",
      f"{(err.pct_error.abs() <= .10).mean():.0%} / {(err.pct_error.abs() <= .20).mean():.0%}")
""")

md_result("failures")

md(r"""
---
# Part 5 — Final model, deployment and reflection

## 5.1 Train the final model on all data and save it for the app
The recommended model is refitted on all 118 sales, with its hyperparameters chosen by 5-fold CV on the full dataset. Alongside the model, I save metadata the app needs to be honest about uncertainty: the cross-validated error, the empirical spread of out-of-fold errors (used to show a likely price range), and the range of each input seen in training (used to warn when a user asks about something the model has never seen).
""")

code(r"""
final_search = sklearn.base.clone(models[best_name]).fit(X, y)
final_model = final_search.best_estimator_
print("Final hyperparameters:", {k.split("__")[-1]: v for k, v in final_search.best_params_.items()})

# Likely range: 10th and 90th percentiles of out-of-fold log errors (log(actual) - log(predicted))
log_resid = np.log(y) - np.log(oof[best_name])
q10, q90 = np.quantile(log_resid, [0.10, 0.90])

meta = {
    "model_name": best_name,
    "trained_on": len(df),
    "trained_at": "2026-10-01",
    "sale_dates": [str(df.sold_date.min().date()), str(df.sold_date.max().date())],
    "cv_mape": float(results.loc[best_name, "val MAPE"]),
    "cv_mae": float(results.loc[best_name, "val MAE"]),
    "interval_log_quantiles": [float(q10), float(q90)],
    "hyperparameters": {k.split("__")[-1]: v for k, v in final_search.best_params_.items()},
    "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in ["beds", "baths", "parking", "land_area_m2"]},
    "segment_counts": {f"{s}_{c}": int(n) for (s, c), n in df.groupby(["suburb", "dwelling_class"]).size().items()},
    "sklearn_version": sklearn.__version__,
}
Path("app/model").mkdir(parents=True, exist_ok=True)
joblib.dump(final_model, "app/model/house_price_model.joblib")
Path("app/model/model_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps({k: meta[k] for k in ["model_name", "trained_on", "cv_mape", "interval_log_quantiles"]}, indent=2))
""")

code(r"""
# Smoke test: reload the saved model exactly as the app does and price two example properties
loaded = joblib.load("app/model/house_price_model.joblib")
examples = pd.DataFrame([
    {"suburb": "Penrith", "property_type": "House", "address": "", "beds": 3, "baths": 1, "parking": 1,
     "area_m2": 600, "sale_method": "private treaty", "sold_date": "2026-09-15"},
    {"suburb": "Bondi", "property_type": "Apartment / Unit / Flat", "address": "", "beds": 2, "baths": 1, "parking": 1,
     "area_m2": None, "sale_method": "auction", "sold_date": "2026-09-15"},
])
pred = loaded.predict(hf.engineer_features(examples))
for (_, r), p in zip(examples.iterrows(), pred):
    print(f"{r.suburb} {r.property_type}, {r.beds} bed: {money(p)}  (likely range {money(p*np.exp(q10))} to {money(p*np.exp(q90))})")
""")

md(r"""
## 5.2 The web application
The app (`app/app.py`) is a small **Flask** application. It imports the same `housing_features.py` used here, loads `house_price_model.joblib`, and offers two ways in:

1. **Single property form.** Choose suburb and property type, enter bedrooms, bathrooms, parking, optional land size, sale method and expected sale month. The app returns the predicted price, a likely range (10th–90th percentile of the model's cross-validated errors), and warnings when inputs fall outside what the model was trained on.
2. **CSV upload.** Upload a CSV with the same columns (a template is downloadable from the app) to price many properties at once. Results show on screen and download as CSV.

Run it with `python app/app.py` and open http://127.0.0.1:5000. Screenshots and full instructions are in the report and `README.md`.
""")

md_result("reflection")
md_result("genai")

# =========================================================================
nb = {"cells": [], "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, (kind, src) in enumerate(CELLS):
    lines = src.splitlines(keepends=True)
    cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": lines}
    if kind == "code":
        cell.update(outputs=[], execution_count=None)
    nb["cells"].append(cell)

best = RESULTS.get("best_model", "Gradient Boosting")
text = json.dumps(nb, indent=1).replace("BEST_MODEL_PLACEHOLDER", best)
text = text.replace('best_name = RESULTS_BEST = \\"' + best + '\\"', 'best_name = \\"' + best + '\\"  # recommended in Part 3')
out = Path(__file__).resolve().parent.parent / "housing_price_prediction.ipynb"
out.write_text(text)
print(f"Wrote {out} with {len(CELLS)} cells")
