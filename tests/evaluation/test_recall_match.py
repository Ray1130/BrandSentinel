from datetime import date

import polars as pl
import pytest

from brandsentinel.core.schemas import empty_table
from brandsentinel.core.types import Table
from brandsentinel.evaluation.recall_match import match_recall_labels


def test_matches_only_manually_verified_exact_asins(mock_tables):
    recalls = pl.DataFrame(
        {
            "asin": [" b0mock0000 ", "B0NOTINREVIEWS", "B0MOCK0001"],
            "recall_id": ["CPSC-EXAMPLE-1", "CPSC-EXAMPLE-2", "CPSC-EXAMPLE-3"],
            "recall_date": [date(2025, 1, 1)] * 3,
            "source": ["CPSC"] * 3,
            "source_url": ["https://www.cpsc.gov/Recalls/example"] * 3,
            "product_name": ["Example product"] * 3,
            "manually_verified": [True, True, False],
        }
    )

    labels = match_recall_labels(
        recalls, mock_tables[Table.CLEAN_REVIEWS], category="Mock_Category"
    )

    assert labels.select("sku", "recall_id").rows() == [("B0MOCK0000", "CPSC-EXAMPLE-1")]
    assert labels["match_type"].to_list() == ["exact_asin"]
    assert labels["match_confidence"].to_list() == [1.0]
    assert labels["source_url"].item() == "https://www.cpsc.gov/Recalls/example"


def test_no_matching_review_sku_returns_contract_empty(mock_tables):
    recalls = pl.DataFrame(
        {
            "asin": ["B0NO-MATCH"],
            "recall_id": ["CPSC-EXAMPLE-1"],
            "recall_date": [date(2025, 1, 1)],
            "source": ["CPSC"],
            "source_url": ["https://www.cpsc.gov/Recalls/example"],
            "product_name": ["Example product"],
            "manually_verified": [True],
        }
    )

    labels = match_recall_labels(
        recalls, mock_tables[Table.CLEAN_REVIEWS], category="Mock_Category"
    )

    assert labels.schema == empty_table(Table.RECALL_LABELS).schema
    assert labels.is_empty()


def test_requires_manual_review_flag(mock_tables):
    with pytest.raises(ValueError, match="manually_verified"):
        match_recall_labels(
            pl.DataFrame(
                {
                    "asin": ["B0MOCK0000"],
                    "recall_id": ["CPSC-EXAMPLE-1"],
                    "recall_date": [date(2025, 1, 1)],
                    "source": ["CPSC"],
                    "source_url": ["https://www.cpsc.gov/Recalls/example"],
                    "product_name": ["Example product"],
                }
            ),
            mock_tables[Table.CLEAN_REVIEWS],
            category="Mock_Category",
        )