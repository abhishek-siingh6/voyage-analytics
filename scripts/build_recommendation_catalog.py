"""Build the recommendation catalog consumed by the travel recommender.

Derives a small, deployment-friendly JSON catalog from the raw ``hotels.csv``
and ``users.csv`` datasets:

  1. The hotel catalog: each hotel serves exactly one place at a fixed
     price per day.
  2. Company booking preferences: the relative frequency with which each
     company books each hotel (used for personalisation).

Usage:
    python scripts/build_recommendation_catalog.py \
        --hotels "<dataset>/hotels.csv" \
        --users "<dataset>/users.csv" \
        --output artifacts/hotel_catalog.json
"""

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build the recommendation catalog")
    parser.add_argument(
        "--hotels", default="artifacts/data/hotels.csv", help="Path to hotels.csv"
    )
    parser.add_argument(
        "--users", default="artifacts/data/users.csv", help="Path to users.csv"
    )
    parser.add_argument(
        "--output", default="artifacts/hotel_catalog.json", help="Output JSON path"
    )
    args = parser.parse_args()

    if not os.path.exists(args.hotels):
        raise SystemExit(f"hotels.csv not found at: {args.hotels}")

    hotels = pd.read_csv(args.hotels)
    print(f"Loaded hotels.csv ({len(hotels)} rows)")

    # 1) Hotel catalog : name -> {place, price_per_day}
    catalog_rows = (
        hotels.groupby(["name", "place"], as_index=False)["price"].first()
    )
    hotel_catalog = {
        row["name"]: {
            "place": row["place"],
            "price_per_day": float(row["price"]),
        }
        for _, row in catalog_rows.iterrows()
    }

    # 2) Company booking preferences (frequency of each hotel per company)
    company_prefs = {}
    if os.path.exists(args.users):
        users = pd.read_csv(args.users)
        print(f"Loaded users.csv ({len(users)} rows)")
        joined = hotels.merge(
            users[["code", "company"]],
            left_on="userCode",
            right_on="code",
            how="left",
        )
        joined["company"] = joined["company"].fillna("unknown")
        counts = (
            joined.groupby(["company", "name"]).size().reset_index(name="count")
        )
        for company, grp in counts.groupby("company"):
            total = int(grp["count"].sum())
            prefs = {
                row["name"]: row["count"] / total
                for _, row in grp.iterrows()
            }
            # Normalise to a sorted (score descending) list of hotel -> weight
            company_prefs[company] = {
                k: round(v, 4)
                for k, v in sorted(prefs.items(), key=lambda x: -x[1])
            }

    catalog = {
        "hotels": hotel_catalog,
        "places": sorted({h["place"] for h in hotel_catalog.values()}),
        "company_preferences": company_prefs,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"Wrote catalog to: {args.output}")
    print(f"  hotels: {len(hotel_catalog)}, places: {len(catalog['places'])}")
    print(f"  companies with preferences: {len(company_prefs)}")

    # Emit metadata alongside the catalog for MLflow tracking.
    out_dir = os.path.dirname(args.output) or "."
    metadata = {
        "model_name": "voyage_recommendation",
        "model_version": "1.0",
        "task": "recommendation",
        "framework": "content-based",
        "hotels_data_path": args.hotels,
        "users_data_path": args.users,
        "n_hotels": len(hotel_catalog),
        "n_places": len(catalog["places"]),
        "n_companies": len(company_prefs),
    }
    with open(os.path.join(out_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to: {os.path.join(out_dir, 'model_metadata.json')}")


if __name__ == "__main__":
    main()
