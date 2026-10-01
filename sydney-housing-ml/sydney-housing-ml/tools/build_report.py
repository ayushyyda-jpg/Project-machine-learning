"""Builds report/report.html and renders it to report/SIT307_ML_Mini_Project_Report.pdf (headless Chromium)."""
import html, re
from pathlib import Path
import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
code_features = html.escape((ROOT / "housing_features.py").read_text())
app_src = (ROOT / "app" / "app.py").read_text()
app_core = html.escape(app_src[app_src.index("def predict_frame"):app_src.index("# ---------------------------------------------------------------- routes")].strip())
nb_src = (ROOT / "tools" / "build_notebook.py").read_text()
model_code = html.escape(nb_src[nb_src.index("def preprocessor(scale_numeric):"):nb_src.index('SCORING = {')].strip()
                         + "\n\n# --- the three models (tuned by an inner 5-fold GridSearchCV inside each outer fold) ---\n"
                         + nb_src[nb_src.index("inner = KFold(n_splits=5"):nb_src.index('cv_results = {"Baseline')].strip())

df = pd.read_csv(ROOT / "data" / "sydney_sold_clean.csv")
import sys; sys.path.insert(0, str(ROOT)); import housing_features as hf
df["cls"] = [hf.dwelling_class(t, a) for t, a in zip(df.property_type, df.address)]
seg = df.groupby(["suburb", "cls"]).price.agg(["count", "median", "min", "max"]).reset_index()
seg_rows = "".join(
    f"<tr><td>{r.suburb}</td><td>{r.cls.replace('_', '/')}</td><td class=n>{r['count']}</td>"
    f"<td class=n>${r['median']:,.0f}</td><td class=n>${r['min']:,.0f} – ${r['max']:,.0f}</td></tr>"
    for _, r in seg.iterrows())

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: "Liberation Serif", "Times New Roman", serif; font-size: 10.6pt; line-height: 1.42; color: #111; }
h1 { font-family: "Liberation Sans", Arial, sans-serif; font-size: 19pt; margin: 0 0 2px; }
h2 { break-after: avoid; page-break-after: avoid; font-family: "Liberation Sans", Arial, sans-serif; font-size: 13pt; margin: 18px 0 6px; border-bottom: 1.5px solid #2a6f97; padding-bottom: 2px; color: #1d2733; }
h3 { font-family: "Liberation Sans", Arial, sans-serif; font-size: 11pt; margin: 12px 0 4px; color: #1d2733; }
p { margin: 0 0 7px; text-align: justify; }
ul { margin: 2px 0 8px 18px; padding: 0; } li { margin-bottom: 3px; }
.sub { font-family: "Liberation Sans", Arial, sans-serif; color: #555; font-size: 9.6pt; margin-bottom: 12px; }
table { border-collapse: collapse; width: 100%; font-family: "Liberation Sans", Arial, sans-serif; font-size: 8.6pt; margin: 6px 0 4px; }
th, td { border-bottom: 1px solid #ccc; padding: 3px 5px; text-align: left; vertical-align: top; }
th { background: #eef3f7; } td.n, th.n { text-align: right; white-space: nowrap; } tr.best td { font-weight: bold; background: #f3f8fb; }
figure { margin: 8px 0 10px; page-break-inside: avoid; } figure img { width: 100%; border: 1px solid #e3e3e3; }
figcaption, .cap { font-family: "Liberation Sans", Arial, sans-serif; font-size: 8.6pt; color: #444; margin-top: 3px; }
pre { font-family: "Liberation Mono", monospace; font-size: 7.4pt; line-height: 1.3; background: #f6f8fa; border: 1px solid #e1e4e8; padding: 7px; white-space: pre-wrap; word-break: break-word; }
code { font-family: "Liberation Mono", monospace; font-size: 9pt; background: #f2f2f2; padding: 0 2px; }
.todo { background: #fff3b0; padding: 1px 4px; font-weight: bold; }
.box { border: 1px solid #c9d8e3; background: #f5f9fc; padding: 7px 10px; margin: 8px 0; font-size: 9.8pt; page-break-inside: avoid; }
.two { display: flex; gap: 10px; } .two figure { flex: 1; }
.pb { page-break-before: always; }
"""

BODY = f"""
<h1>Sydney Housing Price Prediction and Decision Support System</h1>
<div class="sub">SIT307 Machine Learning: ML Mini Project (Distinction task) · Raghav · October 2026<br>
Code and data: <span class="todo">[insert GitHub / OneDrive link to the project ZIP here]</span></div>

<section class="body">
<h2>1. Problem definition and data collection</h2>
<p>A real estate agency needs quick, evidence-based price estimates for appraisals and for checking offers: a supervised regression problem with <b>sale price</b> as the target. Because the markets differ so much, I model <b>log(price)</b> and judge models mainly on <b>MAPE</b>, so a $100k miss on a $500k unit counts more than on a $5m house.</p>
<p>I chose three suburbs where I would consider buying, each a distinct market: <b>Bondi</b> (beachside, 7 km from the CBD, very high land values), <b>Parramatta</b> (Sydney's second CBD, high-rise apartments plus older houses on large blocks) and <b>Penrith</b> (outer west, the most affordable, newer apartment precincts and family houses). I expected location to dominate, then dwelling type (a house includes its land), then size.</p>
<p>I collected sales from Domain.com.au's public sold-listing pages, filtered to disclosed prices, recording price, date, sale method, bedrooms, bathrooms, parking, area, property type and listing URL. Spot checks against four individual listings matched exactly. Of 125 collected sales, I removed one duplicate listing and six retirement-village units (leasehold, not market prices), leaving <b>118 sales</b> (Bondi 38, Parramatta 38, Penrith 42; Sep 2025 – Sep 2026).</p>
<p><b>Quality and bias.</b> Area was missing for 60% of records, and for apartments it sometimes recorded the whole building's site (2,364 m² for a 2-bed flat). Domain labelled some apartments as "House". About half of all sales withhold their price, so the data may under-represent disappointing and prestige results. I also pulled house-filtered pages to balance the mix, so suburb medians here are not market medians, and older records are mostly houses. Condition, views and floor level are not captured at all.</p>
</section>

<table><tr><th>Suburb</th><th>Dwelling class</th><th class=n>Sales</th><th class=n>Median</th><th class=n>Range</th></tr>{seg_rows}</table>
<div class="cap">Table 1. Cleaned dataset by suburb and dwelling class (houses include semis and duplexes; apartments labelled "House" are reclassified as units).</div>

<figure><img src="../figures/fig2_suburb_type.png"><figcaption>Figure 1. Price by suburb and dwelling type (log scale). The median house sells for about 2.6–2.7× the median unit in Bondi and Parramatta, but only 1.8× in Penrith.</figcaption></figure>

<section class="body">
<h2>2. Data understanding and feature engineering</h2>
<p>Prices are strongly right-skewed (skewness 1.81, falling to 0.57 after a log transform) and form separate clusters by suburb and type. Within each suburb, price rises steadily with bedrooms (Spearman 0.82–0.89). After adjusting for suburb and type, prices were essentially flat over the window (−1.8% a year). Six robust-z outliers (e.g. a hotel apartment, a dual-occupancy block) were genuine sales, so I kept them.</p>
<p><b>Hypothesis, stated before engineering:</b> the three strongest drivers are <b>suburb</b>, <b>dwelling type</b> and <b>bedrooms</b>. I engineered: a dwelling class that corrects Domain's mislabels; a <b>suburb × class</b> interaction, since the house premium differs by suburb; land area for land-holding dwellings only, plus a "known" flag; baths per bedroom; an auction flag; and months since the start of the window. I left out distance to the CBD because, with only three suburbs, it just relabels the suburb. With the same Ridge model, these features cut MAPE from 18.8% to <b>15.0%</b>, which matches my view that location and type, <i>and how they combine</i>, set price.</p>

<h2>3. Model development and evaluation</h2>
<p>I compared three approaches: <b>Ridge regression</b> (linear and stable, but cannot bend to every segment), <b>Random Forest</b> (bagging; captures interactions, averages away variance) and <b>Gradient Boosting</b> (boosting; often most accurate, but the easiest to overfit). Before training, I predicted the Random Forest would be slightly best, Ridge close behind, and Gradient Boosting the most overfit. I used <b>nested cross-validation</b>: 5-fold CV repeated 3 times, with hyperparameters tuned by an inner 5-fold grid search inside each outer fold, against a segment-median baseline.</p>
</section>

<table>
<tr><th>Model</th><th class=n>Train MAPE</th><th class=n>Validation MAPE</th><th class=n>Validation MAE</th><th class=n>Validation RMSE</th><th class=n>Validation R²</th></tr>
<tr><td>Baseline: segment median</td><td class=n>16.1%</td><td class=n>17.5% ± 2.1</td><td class=n>$284,539</td><td class=n>$462,662</td><td class=n>0.855</td></tr>
<tr><td>Ridge regression</td><td class=n>12.9%</td><td class=n>15.2% ± 2.1</td><td class=n>$245,504</td><td class=n>$388,217</td><td class=n>0.891</td></tr>
<tr><td>Random Forest</td><td class=n>6.9%</td><td class=n>14.7% ± 2.7</td><td class=n>$234,012</td><td class=n>$372,790</td><td class=n>0.903</td></tr>
<tr class="best"><td>Gradient Boosting</td><td class=n>7.9%</td><td class=n>14.6% ± 2.1</td><td class=n>$228,312</td><td class=n>$351,709</td><td class=n>0.908</td></tr>
</table>
<div class="cap">Table 2. Nested cross-validation, 15 outer folds (mean ± SD). Prices are back-transformed from log scale.</div>

<figure><img src="../figures/fig6_complexity.png"><figcaption>Figure 2. Validation curves: training vs validation MAPE as each model's complexity changes.</figcaption></figure>

<section class="body">
<p>All models beat the baseline (Ridge in 14 of 15 folds), but the differences <i>between</i> models are within noise: Gradient Boosting beats Ridge in only 7 of 15 paired folds. <b>Ridge</b> slightly underfits, with a small train–validation gap. Both tree models nearly memorise the training folds: the <b>Random Forest</b> has the largest gap (6.9% vs 14.7%), and <b>Gradient Boosting</b> shows the classic curve, underfitting at 10–25 trees, best near 100, then overfitting as training error falls to 3.7%. Permutation importance confirmed suburb and dwelling type as the top drivers. Bedrooms mattered, but shared their size signal with bathrooms. Land area added little because it was mostly missing. My predictions were therefore partly right: Ridge was competitive, but Gradient Boosting overfit less than the forest.</p>
<p><b>Recommendation: Gradient Boosting.</b> It is statistically tied on MAPE, but it has the lowest RMSE (about 9% below Ridge), so it makes fewer large dollar misses on high-value homes, and it is clearly better on Parramatta houses (13.2% vs 17.9%). The cost is interpretability, and the fact that it cannot extrapolate beyond the prices it has seen.</p>

<h2>4. Investigating prediction failures</h2>
<p>I read each of the five largest out-of-fold errors against its listing page. Four are under-predictions at the top of the market, and each is driven by information the model never sees:</p>
</section>

<table>
<tr><th>#</th><th>Property</th><th>Features</th><th class=n>Actual</th><th class=n>Predicted</th><th class=n>Error</th><th>Explanation from the listing</th></tr>
<tr><td>1</td><td>344 Birrell St, Bondi</td><td>House 5/3/1</td><td class=n>$5.86m</td><td class=n>$3.42m</td><td class=n>−42%</td><td>Newly completed home; only three other 5+ bed Bondi houses in the data</td></tr>
<tr><td>2</td><td>62 Watson St, Bondi</td><td>House 3/2/1</td><td class=n>$3.30m</td><td class=n>$4.92m</td><td class=n>+49%</td><td>Actually a semi-detached character home, treated as a freestanding house</td></tr>
<tr><td>3</td><td>5601/34 Wellington St, Bondi</td><td>Unit 2/2/1</td><td class=n>$2.60m</td><td class=n>$1.61m</td><td class=n>−38%</td><td>Top floor, north-facing, harbour and beach views, new luxury building</td></tr>
<tr><td>4</td><td>124 Thomas St, Parramatta</td><td>House 3/1/1, 739 m²</td><td class=n>$2.30m</td><td class=n>$1.36m</td><td class=n>−41%</td><td>Riverside development site; value is land and zoning, not the house</td></tr>
<tr><td>5</td><td>11 Stephen St, Bondi</td><td>House 3/1/1</td><td class=n>$4.10m</td><td class=n>$3.21m</td><td class=n>−22%</td><td>Original freestanding cottage; one bathroom signals renovation upside, not low value</td></tr>
</table>
<div class="cap">Table 3. Five largest absolute out-of-fold errors (Gradient Boosting). Features are beds/baths/parking.</div>

<section class="body">
<p>Ranked by percentage, the worst cases are cheap and unusual: the hotel apartment (+50%) and a 1-bed walk-up with no parking (+52%). Overall, 76% of sales were predicted within ±20%. The missing information is <b>floor area, condition and age, views and floor level, zoning, attached vs freestanding</b>, and the agent's description, which explained every failure above. Predictions should be treated cautiously for prestige, new-build, development, serviced and rare property types, and outside these suburbs and dates. Homogeneous stock (Penrith apartments, 7–8% error) is inherently easier than heterogeneous Bondi stock (15–20%).</p>

<h2>5. Deployment and reflection</h2>
<p>The trained pipeline is saved with <code>joblib</code>, with metadata: cross-validated error, the 10th–90th percentile of out-of-fold errors (for a likely range) and the training ranges. I built the app in <b>Flask</b>, which the brief allows: Streamlit could not be installed in my build environment, and Flask let me run, test and screenshot the app. It imports the same <code>housing_features.py</code> as the notebook, so features are computed identically. It offers a single-property form, CSV upload with downloadable results, and a JSON API. Every estimate comes with a likely range, the three most similar real sales, and caution flags (Figures 3–4; instructions in Appendix A).</p>
<p><b>Reflection.</b> Data work changed the results more than model choice (about 4 MAPE points from features, under 1 between models), and the baseline showed ML adds roughly 3 points. On small data, nested, repeated CV was essential: a single split could have crowned any model. In deployment, consistent features and honest uncertainty mattered as much as accuracy, and Ridge would be easier to explain. On bias and fairness: the sample over-represents disclosed sales; the model under-values premium homes and over-values cheap atypical units, which would compress real price differences if used for pricing or lending; and suburb, the strongest feature, correlates with income and demographics, so the model can reinforce existing geographic price gaps. It should support, not replace, a valuer. With more resources, I would use the NSW Valuer General's bulk sales data (including withheld prices), more suburbs with geocoded distances, text features from agent descriptions, floor area and zoning, conformal prediction intervals, and a wider tuning search.</p>
</section>

<figure><img src="screenshots/03_single_result.png" style="width:86%;margin:0 7%"><figcaption>Figure 3. Single-property estimate: a 4-bed Parramatta house on 750 m². The app shows the likely range, comparable sales and a large-block caution flag.</figcaption></figure>
<figure><img src="screenshots/04_batch_result.png" style="width:86%;margin:0 7%"><figcaption>Figure 4. CSV upload: six properties priced at once, with caution flags for a thin segment and a 7-bedroom house outside the training range.</figcaption></figure>

<div class="box"><b>GenAI acknowledgement.</b> Claude (Anthropic) was used as an assistant to help read Domain sold-listing pages and transcribe them into the dataset (records were spot-checked against the original listings), to draft and debug the Python code for the notebook and Flask app, and to draft explanatory text. All design choices, results and interpretations were reviewed by me, and I am responsible for the final submission.</div>

<h2>References</h2>
<ul style="font-size:9.6pt">
<li>Domain.com.au (2026) <i>Sold listings: Bondi NSW 2026, Parramatta NSW 2150, Penrith NSW 2750</i>, accessed 30 Sep – 1 Oct 2026. Individual listing URLs are in <code>data/sydney_sold_raw.csv</code>.</li>
<li>Pedregosa, F. et al. (2011) 'Scikit-learn: Machine learning in Python', <i>Journal of Machine Learning Research</i>, 12, pp. 2825–2830.</li>
<li>Breiman, L. (2001) 'Random forests', <i>Machine Learning</i>, 45(1), pp. 5–32.</li>
<li>Friedman, J.H. (2001) 'Greedy function approximation: a gradient boosting machine', <i>Annals of Statistics</i>, 29(5), pp. 1189–1232.</li>
<li>Hoerl, A.E. and Kennard, R.W. (1970) 'Ridge regression: biased estimation for nonorthogonal problems', <i>Technometrics</i>, 12(1), pp. 55–67.</li>
<li>Varma, S. and Simon, R. (2006) 'Bias in error estimation when using cross-validation for model selection', <i>BMC Bioinformatics</i>, 7, 91.</li>
<li>NSW Valuer General (2026) <i>Bulk property sales information</i>, https://www.valuergeneral.nsw.gov.au (suggested data source for future work).</li>
</ul>

<h2 class="pb">Appendix A. How to build, run and use the code and app</h2>
<p><b>Setup</b> (Python 3.10+): unzip the project (or clone the repository), then in the project folder run <code>python -m venv .venv</code>, activate it, and run <code>pip install -r requirements.txt</code>.</p>
<p><b>Dataset:</b> <code>python data/build_dataset.py</code> merges the raw collection batches into <code>data/sydney_sold_raw.csv</code> (already included).</p>
<p><b>Notebook:</b> open <code>housing_price_prediction.ipynb</code> and choose <i>Kernel → Restart &amp; Run All</i>, or run <code>jupyter nbconvert --to notebook --execute --inplace housing_price_prediction.ipynb</code>. It takes 3–5 minutes and writes the cleaned data, figures and the model to <code>app/model/</code>. All randomness is seeded (42).</p>
<p><b>Web app:</b> run <code>python app/app.py</code> and open <code>http://127.0.0.1:5000</code>.</p>
<ul>
<li><i>Single property:</i> choose suburb and property type; enter bedrooms, bathrooms, parking, optional land size (houses, townhouses and villas only), sale method and expected sale month; press <b>Estimate price</b>. Read the estimate together with its likely range, the comparable sales and any caution flags.</li>
<li><i>Upload CSV:</i> click the <b>Upload CSV</b> tab, download the template (or use <code>app/example_properties.csv</code>), fill in one property per row (<code>suburb, property_type, beds, baths, parking, area_m2, sale_method, sold_date</code>), upload it and press <b>Predict prices</b>; then use <b>Download results as CSV</b>.</li>
<li><i>API:</i> <code>POST /api/predict</code> with a JSON object (or list) using the same fields.</li>
</ul>
<div class="two"><figure><img src="screenshots/01_home.png"><figcaption>Figure A1. App start page.</figcaption></figure>
<figure><img src="screenshots/02_form_filled.png"><figcaption>Figure A2. Form filled in before pressing Estimate price.</figcaption></figure></div>

<h2 class="pb">Appendix B. Key code</h2>
<h3>B1. Feature engineering shared by notebook and app (<code>housing_features.py</code>)</h3>
<pre>{code_features}</pre>
<h3>B2. Model pipelines and nested cross-validation (from the notebook)</h3>
<pre>{model_code}</pre>
<h3>B3. Prediction in the app (<code>app/app.py</code>)</h3>
<pre>{app_core}</pre>
<p class="cap">The complete code, all outputs and every figure are in the executed notebook <code>housing_price_prediction.ipynb</code> and the project repository.</p>
"""

page = f"<!doctype html><html><head><meta charset='utf-8'><title>SIT307 ML Mini Project Report</title><style>{CSS}</style></head><body>{BODY}</body></html>"
(REPORT / "report.html").write_text(page)

words = sum(len(re.sub("<.*?>", " ", s).split()) for s in re.findall(r'<section class="body">(.*?)</section>', BODY, flags=re.S))
print(f"Body word count (excluding tables, figures, captions, references, appendices): {words}")

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto((REPORT / "report.html").as_uri())
    pg.wait_for_load_state("networkidle")
    pg.pdf(path=str(REPORT / "SIT307_ML_Mini_Project_Report.pdf"), format="A4", print_background=True,
           display_header_footer=True, header_template="<span></span>",
           footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#777'>SIT307 ML Mini Project · page <span class='pageNumber'></span> of <span class='totalPages'></span></div>",
           margin={"top": "16mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
    b.close()
print("PDF written")
