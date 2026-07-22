import pandas as pd
import pytest

from quant_lab.portfolio.sizing import (
    scores_to_group_neutral_long_only_capped_weights,
    scores_to_group_neutral_long_only_weights,
)


def test_group_neutral_long_only_allocates_equal_group_gross():
    scores = pd.DataFrame({"a": [1.0], "b": [3.0], "c": [2.0], "d": [4.0]})
    groups = pd.Series({"a": "x", "b": "x", "c": "y", "d": "y"})

    weights = scores_to_group_neutral_long_only_weights(scores, groups, top_fraction=0.5)

    assert weights.loc[0, ["a", "b"]].sum() == 0.5
    assert weights.loc[0, ["c", "d"]].sum() == 0.5
    assert weights.loc[0, "b"] > weights.loc[0, "a"]
    assert weights.loc[0, "d"] > weights.loc[0, "c"]


def test_group_neutral_long_only_capped_allocates_cash_when_group_capacity_is_thin():
    scores = pd.DataFrame({"a": [1.0], "b": [3.0], "c": [2.0], "d": [4.0], "e": [5.0]})
    groups = pd.Series({"a": "x", "b": "x", "c": "y", "d": "y", "e": "y"})

    weights = scores_to_group_neutral_long_only_capped_weights(
        scores,
        groups,
        top_fraction=1.0,
        max_weight=0.20,
    )

    assert weights.loc[0].max() <= 0.20
    assert weights.loc[0, ["a", "b"]].sum() == pytest.approx(0.40)
    assert weights.loc[0, ["c", "d", "e"]].sum() == pytest.approx(0.50)
