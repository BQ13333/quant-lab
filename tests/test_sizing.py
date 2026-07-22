import pandas as pd
import pytest

from quant_lab.portfolio.sizing import scores_to_long_only_capped_weights, scores_to_long_only_weights


def test_scores_to_long_only_weights_selects_top_names():
    scores = pd.DataFrame({"a": [1.0], "b": [3.0], "c": [2.0], "d": [-1.0]})
    weights = scores_to_long_only_weights(scores, top_fraction=0.5)

    assert weights.loc[0, "a"] == 0.0
    assert weights.loc[0, "d"] == 0.0
    assert weights.loc[0, ["b", "c"]].sum() == 1.0


def test_scores_to_long_only_capped_weights_can_hold_cash_when_capacity_is_thin():
    scores = pd.DataFrame({"a": [1.0], "b": [4.0], "c": [3.0], "d": [2.0]})

    weights = scores_to_long_only_capped_weights(scores, top_fraction=0.5, max_weight=0.30)

    assert weights.loc[0, ["a", "d"]].sum() == 0.0
    assert weights.loc[0, ["b", "c"]].max() <= 0.30
    assert weights.loc[0].sum() == pytest.approx(0.60)


def test_scores_to_long_only_capped_weights_redistributes_when_capacity_is_available():
    scores = pd.DataFrame({"a": [1.0], "b": [4.0], "c": [3.0], "d": [2.0]})

    weights = scores_to_long_only_capped_weights(scores, top_fraction=1.0, max_weight=0.40)

    assert weights.loc[0].max() <= 0.40
    assert weights.loc[0].sum() == pytest.approx(1.0)
