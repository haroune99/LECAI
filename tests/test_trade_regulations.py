import pytest
from src.tools.trade_regulations import init_kb, trade_regulations_lookup


def test_sanctions_check_clean_entity():
    init_kb()
    result = trade_regulations_lookup(
        query_type="sanctions_check",
        entity_name="FakeNonExistentCompanyXYZ",
    )
    assert result.status == "success"
    assert result.content.get("sanctioned") == False


def test_tariff_missing_params():
    result = trade_regulations_lookup(query_type="tariff")
    assert result.status == "error"
