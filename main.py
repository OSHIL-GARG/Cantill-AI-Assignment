
"""
Main entry point for the Cantill Spend Intelligence project.

Examples:
    python main.py --company A
    python main.py --company B
    python main.py --company C

Custom configuration:
    python main.py --company C --config configs/company_c.yaml

Custom log directory:
    python main.py --company A --log-dir logs/company_a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.pipeline.run_pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_COMPANIES = ("A", "B", "C")


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cantill Spend Intelligence - Company Spend Processing"
    )

    parser.add_argument(
        "--company",
        type=str.upper,
        choices=SUPPORTED_COMPANIES,
        required=True,
        help="Company to process: A, B, or C.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a company YAML configuration file.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional directory for pipeline log files.",
    )

    return parser.parse_args()


def get_config_path(args: argparse.Namespace) -> Path:
    """Get the selected YAML configuration path."""
    if args.config:
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
    else:
        config_path = (
            PROJECT_ROOT
            / "configs"
            / f"company_{args.company.lower()}.yaml"
        )

    config_path = config_path.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    return config_path


def print_summary(
    company: str,
    output_path: Path,
    result: pd.DataFrame,
) -> None:
    """Print a concise summary after successful processing."""
    included = int((result["include"] == 1).sum())
    excluded = int((result["include"] == 0).sum())

    total_spend = pd.to_numeric(
        result["spend"], errors="coerce"
    ).fillna(0).sum()

    print("\n" + "=" * 60)
    print("CANTILL SPEND INTELLIGENCE")
    print("=" * 60)

    print(f"Company             : {company}")
    print(f"Total records       : {len(result):,}")
    print(f"Included records    : {included:,}")
    print(f"Excluded records    : {excluded:,}")
    print(f"Total included spend: {total_spend:,.2f}")
    print(f"Output file         : {output_path}")

    print("\nExclusion summary:")

    excluded_rows = result.loc[result["include"] == 0]

    if excluded_rows.empty:
        print("No excluded records.")
    else:
        reasons = (
            excluded_rows["exclusion_reason"]
            .fillna("UNKNOWN")
            .value_counts()
        )

        for reason, count in reasons.items():
            print(f"  {reason}: {count:,}")

    print("\nPipeline completed successfully.")
    print("=" * 60)


def main() -> int:
    """Execute the selected company pipeline."""
    args = parse_arguments()

    try:
        config_path = get_config_path(args)

        log_directory = None
        if args.log_dir:
            log_directory = Path(args.log_dir).expanduser()
            if not log_directory.is_absolute():
                log_directory = PROJECT_ROOT / log_directory

        print("\nStarting Cantill Spend Intelligence...")
        print(f"Selected company: {args.company}")
        print(f"Configuration   : {config_path}")

        output_path, result = run_pipeline(
            config_path=config_path,
            log_directory=log_directory,
        )

        print_summary(
            company=args.company,
            output_path=output_path,
            result=result,
        )

        return 0

    except FileNotFoundError as exc:
        print(f"\nFILE ERROR: {exc}", file=sys.stderr)
        return 1

    except (ValueError, KeyError, TypeError) as exc:
        print(f"\nCONFIGURATION OR DATA ERROR: {exc}", file=sys.stderr)
        return 1

    except Exception as exc:
        print(
            f"\nPIPELINE FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
