
"""Evaluate basic output quality for Companies A and B."""

from pathlib import Path

import pandas as pd

from src.utils import OUTPUT_COLUMNS

ROOT = Path(__file__).resolve().parent
SUBMISSION_DIR = ROOT / "submission"

FILES = {
    "A": "Oshil_Garg_A.csv",
    "B": "Oshil_Garg_B.csv",
}


def evaluate_company(company, filename):
    path = SUBMISSION_DIR / filename

    if not path.exists():
        print(f"Company {company}: output file not found.")
        return

    df = pd.read_csv(path)

    missing = [
        column for column in OUTPUT_COLUMNS
        if column not in df.columns
    ]

    if missing:
        print(f"Company {company}: missing columns {missing}")
        return

    included = df[df["include"] == 1]
    excluded = df[df["include"] == 0]

    print("\n" + "=" * 50)
    print(f"COMPANY {company} OUTPUT SUMMARY")
    print("=" * 50)

    print(f"Total records: {len(df):,}")
    print(f"Included: {len(included):,}")
    print(f"Excluded: {len(excluded):,}")
    print(f"Included spend: {included['spend'].sum():,.2f}")

    print("\nExclusion reasons:")
    print(
        excluded["exclusion_reason"]
        .fillna("UNKNOWN")
        .value_counts()
        .to_string()
    )

    print("\nClassification coverage:")
    for column in ["l1", "l2", "l3", "l4"]:
        coverage = (
            included[column].fillna("").astype(str).str.strip().ne("")
        ).mean() * 100

        print(f"{column.upper()} populated: {coverage:.2f}%")


def main():
    for company, filename in FILES.items():
        evaluate_company(company, filename)


if __name__ == "__main__":
    main()
