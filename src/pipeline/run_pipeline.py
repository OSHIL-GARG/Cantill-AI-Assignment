
"""
Main pipeline orchestrator for the Cantill Spend Intelligence project.

Responsibilities:
1. Load the company YAML configuration.
2. Resolve input and output paths.
3. Validate required input files.
4. Execute the correct company-specific engine.
5. Validate the generated output.
6. Save the final submission CSV.
7. Record pipeline execution details in a log.

Company processing logic remains in src/engine.py.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..engine import run_a, run_b, run_c
from ..output.generate_output import write_output
from ..utils import OUTPUT_COLUMNS


# ---------------------------------------------------------------------
# Project paths and company configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPANY_RUNNERS = {
    "A": run_a,
    "B": run_b,
    "C": run_c,
}

REQUIRED_INPUTS = {
    "A": ("erp", "tool", "taxonomy", "labels"),
    "B": ("invoices", "fx", "taxonomy", "labels"),
    "C": ("invoices", "taxonomy"),
}


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_logger(
    log_directory: str | Path | None = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a project logger with console and file output.

    A timestamped log file is created in the logs directory.
    Existing root logger configuration is not modified.
    """
    directory = (
        Path(log_directory)
        if log_directory is not None
        else PROJECT_ROOT / "logs"
    )

    directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cantill_pipeline")
    logger.setLevel(log_level)
    logger.propagate = False

    # Avoid adding duplicate handlers if called more than once.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = directory / f"pipeline_{timestamp}.log"

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------
# Configuration loading and path resolution
# ---------------------------------------------------------------------

def _resolve_project_path(
    value: str | Path,
    project_root: Path,
) -> Path:
    """Resolve a configured path relative to the project root."""

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def load_pipeline_config(
    config_path: str | Path,
) -> tuple[dict[str, Any], Path]:
    """
    Load and validate a company YAML configuration.

    Relative input and output paths are resolved from the project root.

    Returns
    -------
    config:
        Configuration dictionary with absolute file paths.
    project_root:
        Root directory used to resolve relative paths.
    """
    config_file = Path(config_path).expanduser()

    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}"
        )

    if not config_file.is_file():
        raise ValueError(
            f"Configuration path is not a file: {config_file}"
        )

    # The supplied project configurations live in <project>/configs/.
    if config_file.parent.name.lower() == "configs":
        project_root = config_file.parent.parent
    else:
        project_root = PROJECT_ROOT

    try:
        with config_file.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML configuration: {config_file}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    company = str(config.get("company", "")).strip().upper()

    if company not in COMPANY_RUNNERS:
        raise ValueError(
            f"Unsupported company '{company}'. "
            f"Supported companies: {list(COMPANY_RUNNERS)}"
        )

    config["company"] = company

    inputs = config.get("inputs")

    if not isinstance(inputs, dict):
        raise ValueError(
            "Configuration must contain an 'inputs' mapping."
        )

    # Validate company-specific input keys.
    missing_keys = [
        key
        for key in REQUIRED_INPUTS[company]
        if not inputs.get(key)
    ]

    if missing_keys:
        raise ValueError(
            f"Company {company} configuration is missing inputs: "
            f"{missing_keys}"
        )

    # Resolve all configured input paths.
    resolved_inputs = {}

    for name, value in inputs.items():
        if value is None or str(value).strip() == "":
            resolved_inputs[name] = None
        else:
            resolved_inputs[name] = str(
                _resolve_project_path(value, project_root)
            )

    config["inputs"] = resolved_inputs

    output_value = config.get("output")

    if not output_value:
        raise ValueError(
            "Configuration must specify an output CSV path."
        )

    config["output"] = str(
        _resolve_project_path(output_value, project_root)
    )

    config["_project_root"] = str(project_root)
    config["_config_file"] = str(config_file)

    return config, project_root


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def validate_input_files(
    config: dict[str, Any],
) -> None:
    """
    Check that all required input files exist before processing.

    Company C's FX file is optional in the engine. If configured,
    it is used when it exists; a missing optional FX file is logged
    later as a warning.
    """
    company = config["company"]
    inputs = config["inputs"]

    missing_files = []

    for key in REQUIRED_INPUTS[company]:
        value = inputs.get(key)

        if not value:
            missing_files.append(f"{key}: path not configured")
            continue

        path = Path(value)

        if not path.exists():
            missing_files.append(f"{key}: {path}")
        elif not path.is_file():
            missing_files.append(f"{key}: not a file ({path})")

    if missing_files:
        details = "\n".join(
            f" - {item}" for item in missing_files
        )

        raise FileNotFoundError(
            "Required input files are missing:\n" + details
        )

    # Company C may run without an FX file if all records are USD
    # or have valid row-level rates.
    if company == "C" and inputs.get("fx"):
        fx_path = Path(inputs["fx"])

        if not fx_path.exists():
            logging.getLogger("cantill_pipeline").warning(
                "Optional Company C FX file does not exist: %s",
                fx_path,
            )


# ---------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------

def validate_pipeline_output(
    df: pd.DataFrame,
) -> None:
    """
    Validate the final submission DataFrame.

    Checks:
    - Required output columns are present.
    - No output rows are missing record IDs.
    - include contains only 0 or 1.
    - spend contains finite numeric values.
    - confidence is between 0 and 1.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Pipeline engine must return a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Pipeline generated an empty output DataFrame."
        )

    missing_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Pipeline output is missing columns: {missing_columns}"
        )

    if df["record_id"].isna().any():
        raise ValueError("Output contains missing record IDs.")

    if df["record_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Output contains blank record IDs.")

    include = pd.to_numeric(
        df["include"],
        errors="coerce",
    )

    if include.isna().any() or not include.isin([0, 1]).all():
        raise ValueError(
            "The include column must contain only 0 or 1."
        )

    spend = pd.to_numeric(
        df["spend"],
        errors="coerce",
    )

    if spend.isna().any():
        raise ValueError(
            "The spend column contains missing or non-numeric values."
        )

    if not spend.map(math.isfinite).all():
        raise ValueError(
            "The spend column contains infinite values."
        )

    confidence = pd.to_numeric(
        df["confidence"],
        errors="coerce",
    )

    if confidence.isna().any():
        raise ValueError(
            "The confidence column contains missing or non-numeric values."
        )

    if not confidence.between(0, 1).all():
        raise ValueError(
            "Confidence values must be between 0 and 1."
        )


# ---------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------

def run_pipeline(
    config_path: str | Path,
    log_directory: str | Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """
    Execute the complete company-specific spend pipeline.

    Parameters
    ----------
    config_path:
        Path to company_a.yaml, company_b.yaml or company_c.yaml.
    log_directory:
        Optional directory for pipeline logs.

    Returns
    -------
    output_path:
        Path to the generated submission CSV.
    result:
        Final validated pandas DataFrame.
    """
    logger = setup_logger(log_directory)

    start_time = datetime.now()

    logger.info("=" * 65)
    logger.info("CANTILL SPEND INTELLIGENCE PIPELINE STARTED")
    logger.info("=" * 65)

    try:
        # 1. Load configuration.
        config, project_root = load_pipeline_config(config_path)

        company = config["company"]
        logger.info("Company selected: %s", company)
        logger.info("Configuration: %s", config["_config_file"])
        logger.info("Project root: %s", project_root)

        # 2. Validate source files.
        logger.info("Validating input files...")
        validate_input_files(config)
        logger.info("All required input files are available.")

        # 3. Select company-specific processing function.
        runner = COMPANY_RUNNERS[company]

        logger.info(
            "Starting Company %s processing using %s...",
            company,
            runner.__name__,
        )

        # 4. Execute company processing engine.
        result = runner(config)

        logger.info(
            "Processing completed. Generated %s records.",
            f"{len(result):,}",
        )

        # 5. Validate generated data.
        logger.info("Validating final output...")
        validate_pipeline_output(result)
        logger.info("Output validation successful.")

        # 6. Save submission CSV.
        output_path = Path(config["output"])
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        saved_path = write_output(
            df=result,
            path=output_path,
        )

        saved_path = Path(saved_path).resolve()

        # 7. Log summary statistics.
        included = int(
            pd.to_numeric(result["include"]).eq(1).sum()
        )

        excluded = len(result) - included

        total_spend = float(
            pd.to_numeric(result["spend"]).sum()
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info("-" * 65)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("Company: %s", company)
        logger.info("Total records: %s", f"{len(result):,}")
        logger.info("Included records: %s", f"{included:,}")
        logger.info("Excluded records: %s", f"{excluded:,}")
        logger.info("Total included spend: %.2f", total_spend)
        logger.info("Output saved: %s", saved_path)
        logger.info("Execution time: %.2f seconds", elapsed)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 65)

        return saved_path, result

    except Exception:
        logger.exception("PIPELINE EXECUTION FAILED")
        raise


# ---------------------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Cantill Spend Intelligence pipeline."
    )

    parser.add_argument(
        "--company",
        choices=["A", "B", "C"],
        required=True,
        help="Company to process.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration path.",
    )

    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional directory for execution logs.",
    )

    args = parser.parse_args()

    selected_config = args.config or (
        PROJECT_ROOT / "configs" / f"company_{args.company.lower()}.yaml"
    )

    output_file, output_df = run_pipeline(
        config_path=selected_config,
        log_directory=args.log_dir,
    )

    print("\nPipeline completed successfully.")
    print(f"Output file: {output_file}")
    print(f"Total records: {len(output_df):,}")
    print(
        "Included records:",
        int(output_df["include"].eq(1).sum()),
    )
    print(
        "Excluded records:",
        int(output_df["include"].eq(0).sum()),
    )
