
import pandas as pd
import pytest

from src.utils import (
    OUTPUT_COLUMNS,
    norm_text,
    norm_id,
    parse_amount,
)
from src.output.generate_output import (
    prepare_output,
    validate_output,
)


def test_output_schema():
    assert len(OUTPUT_COLUMNS) == 11


def test_text_normalization():
    assert norm_text("ABC Suppliers Ltd.") == "abc suppliers ltd"


def test_id_normalization():
    assert norm_id("PO-123/45") == "PO12345"


def test_amount_parsing():
    values = pd.Series(["1,250.50", "(500.00)", "100"])
    result = parse_amount(values)

    assert result.tolist() == [1250.50, -500.00, 100.00]


def test_prepare_output():
    data = pd.DataFrame([{
        "record_id": "TEST-001",
        "include": 1,
        "exclusion_reason": "",
        "supplier_key": "SUP-001",
        "item_key": "ITEM-001",
        "spend": 100.50,
        "l1": "IT",
        "l2": "Hardware",
        "l3": "Computers",
        "l4": "Laptops",
        "confidence": 0.95,
    }])

    result = prepare_output(data)

    assert list(result.columns) == OUTPUT_COLUMNS
    assert result.iloc[0]["spend"] == 100.50


def test_excluded_record_spend():
    data = pd.DataFrame([{
        "record_id": "TEST-002",
        "include": 0,
        "exclusion_reason": "DUPLICATE",
        "supplier_key": "SUP-001",
        "item_key": "ITEM-001",
        "spend": 0,
        "l1": "",
        "l2": "",
        "l3": "",
        "l4": "",
        "confidence": 0,
    }])

    assert validate_output(data) is True


def test_invalid_confidence():
    data = pd.DataFrame([{
        "record_id": "TEST-003",
        "include": 1,
        "exclusion_reason": "",
        "supplier_key": "",
        "item_key": "",
        "spend": 100,
        "l1": "",
        "l2": "",
        "l3": "",
        "l4": "",
        "confidence": 1.5,
    }])

    with pytest.raises(ValueError):
        prepare_output(data)
