
"""Configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_COMPANIES = {"A", "B", "C"}

REQUIRED_INPUTS = {
    "A": ["erp", "tool", "taxonomy", "labels"],
    "B": ["invoices", "fx", "taxonomy", "labels"],
    "C": ["invoices", "taxonomy"],
}


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = root / path

    return path.resolve()


def load_config(
    config_path: str | Path,
    check_input_files: bool = False,
) -> dict[str, Any]:
    """
    Load and validate a company YAML configuration.

    Relative input and output paths are resolved relative to the project root.
    """
    config_file = Path(config_path).expanduser()

    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file

    config_file = config_file.resolve()

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_file}"
        )

    if not config_file.is_file():
        raise ValueError(
            f"Configuration path is not a file: {config_file}"
        )

    try:
        with config_file.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML syntax in {config_file}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            "The YAML configuration must contain a mapping."
        )

    company = str(config.get("company", "")).strip().upper()

    if company not in SUPPORTED_COMPANIES:
        raise ValueError(
            "Configuration must specify company A, B, or C."
        )

    config["company"] = company

    inputs = config.get("inputs")

    if not isinstance(inputs, dict):
        raise ValueError(
            "Configuration must contain an 'inputs' mapping."
        )

    missing_keys = [
        key
        for key in REQUIRED_INPUTS[company]
        if not inputs.get(key)
    ]

    if missing_keys:
        raise ValueError(
            f"Company {company} is missing input paths: {missing_keys}"
        )

    resolved_inputs = {}

    for key, value in inputs.items():
        if value is None or str(value).strip() == "":
            resolved_inputs[key] = None
        else:
            resolved_inputs[key] = str(resolve_path(value))

    config["inputs"] = resolved_inputs

    output = config.get("output")

    if not output:
        raise ValueError(
            "Configuration must specify an output file."
        )

    config["output"] = str(resolve_path(output))

    config["reporting_currency"] = str(
        config.get("reporting_currency", "USD")
    ).strip().upper()

    if len(config["reporting_currency"]) != 3:
        raise ValueError(
            "reporting_currency must be a three-letter currency code."
        )

    config["_project_root"] = str(PROJECT_ROOT)
    config["_config_file"] = str(config_file)

    if check_input_files:
        validate_config_inputs(config)

    return config


def validate_config_inputs(config: dict[str, Any]) -> None:
    """Check that all required input files exist."""
    company = config["company"]

    missing = []

    for key in REQUIRED_INPUTS[company]:
        value = config["inputs"].get(key)

        if not value:
            missing.append(f"{key}: no path configured")
            continue

        path = Path(value)

        if not path.exists():
            missing.append(f"{key}: {path}")
        elif not path.is_file():
            missing.append(f"{key}: not a file ({path})")

    if missing:
        raise FileNotFoundError(
            "Required input files are missing:\n"
            + "\n".join(f"- {item}" for item in missing)
        )

    # FX is optional for Company C.
    optional_fx = config["inputs"].get("fx")

    if (
        company == "C"
        and optional_fx
        and not Path(optional_fx).is_file()
    ):
        raise FileNotFoundError(
            f"Configured optional FX file does not exist: {optional_fx}"
        )


# Backward-compatible alias for older project code.
def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper around load_config()."""
    return load_config(config_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate a Cantill company configuration."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to a company YAML file.",
    )

    args = parser.parse_args()

    loaded = load_config(args.config, check_input_files=True)

    print("Configuration loaded successfully.")
    print(f"Company: {loaded['company']}")
    print(f"Reporting currency: {loaded['reporting_currency']}")
    print(f"Output: {loaded['output']}")
