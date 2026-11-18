
"""Validate generated Cantill submission CSV files."""

from pathlib import Path

import pandas as pd

from src.output.generate_output import validate_output

ROOT = Path(__file__).resolve().parent
SUBMISSION_DIR = ROOT / "submission"

FILES = [
    "Oshil_Garg_A.csv",
    "Oshil_Garg_B.csv",
]


def main():
    for filename in FILES:
        path = SUBMISSION_DIR / filename

        if not path.exists():
            print(f"MISSING: {filename}")
            continue

        print(f"\nValidating {filename}...")

        dataframe = pd.read_csv(path)

        validate_output(dataframe)

        print("VALID")
        print(f"Records: {len(dataframe):,}")
        print(f"Included: {(dataframe['include'] == 1).sum():,}")
        print(f"Excluded: {(dataframe['include'] == 0).sum():,}")
        print(f"Total spend: {dataframe['spend'].sum():,.2f}")

    print("\nValidation process completed.")


if __name__ == "__main__":
    main()
