import pytest
from src.tools.trade_calculator import trade_calculator


def test_currency_convert_cny_to_gbp():
    result = trade_calculator(
        operation="currency_convert",
        params={"amount": 100000, "from_currency": "CNY", "to_currency": "GBP"},
    )
    assert result.status == "success"
    assert result.content["result"] > 0


def test_landed_cost_calculation():
    result = trade_calculator(
        operation="landed_cost",
        params={
            "fob_price": 10.0,
            "units": 1000,
            "freight": 500,
            "insurance_rate": 0.5,
            "duty_rate": 12.8,
            "vat_rate": 20,
            "handling": 200,
            "customs_clearance": 150,
        },
    )
    assert result.status == "success"
    assert result.content["total_landed_cost"] > 0
    assert result.content["uk_import_duty"] > 0


def test_roi_projection():
    result = trade_calculator(
        operation="roi_projection",
        params={"principal": 2000000, "annual_rate": 8, "years": 3},
    )
    assert result.status == "success"
    assert result.content["future_value"] > 2000000
