"""
Merge the manually collected Domain.com.au sold-listing batches into one raw CSV.

Each batch was copied from a public Domain "sold listings" results page
(filtered to listings with a disclosed price) between 30 Sep and 1 Oct 2026.
No cleaning happens here: duplicates, odd property-type labels and suspicious
areas are kept on purpose so the notebook can show (and justify) every
cleaning decision.

Run:  python data/build_dataset.py
Out:  data/sydney_sold_raw.csv
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
COLS = ["suburb", "address", "price", "sold_date", "sale_method", "beds",
        "baths", "parking", "area_m2", "property_type", "url"]

# 1) first collection pass (already in tidy form, ISO dates)
first = pd.read_csv(HERE / "raw_listings.csv", sep="|", dtype=str)
first["source_page"] = "sold-listings (all types), page 1"

# 2) later batches: "address, Suburb|$price|dd Mon yyyy|method|..."
batches = {
    "batch_parra_houses.txt":  "sold-listings/house, price disclosed, page 1",
    "batch_parra_units2.txt":  "sold-listings (all types), price disclosed, page 2",
    "batch_bondi_houses.txt":  "sold-listings/house, price disclosed, page 1",
    "batch_bondi_units.txt":   "sold-listings/apartment, price disclosed, page 1",
    "batch_penrith_houses.txt": "sold-listings/house, price disclosed, page 1",
    "batch_penrith_p2.txt":    "sold-listings (all types), price disclosed, page 2",
}
frames = [first]
for fname, page in batches.items():
    b = pd.read_csv(HERE / fname, sep="|", header=None, dtype=str,
                    names=["address_full"] + COLS[2:])
    split = b["address_full"].str.rsplit(", ", n=1, expand=True)
    b["address"], b["suburb"] = split[0], split[1]
    b["price"] = b["price"].str.replace(r"[$,]", "", regex=True)
    b["sold_date"] = pd.to_datetime(b["sold_date"], format="%d %b %Y").dt.strftime("%Y-%m-%d")
    b["source_page"] = page
    frames.append(b.drop(columns="address_full"))

raw = pd.concat(frames, ignore_index=True)[COLS + ["source_page"]]
raw = raw.drop_duplicates(subset="url", keep="first")   # same listing seen on two result pages
raw["collected_on"] = "2026-10-01"
raw.to_csv(HERE / "sydney_sold_raw.csv", index=False)
print(f"{len(raw)} unique listing URLs")
print(raw["suburb"].value_counts().to_string())
