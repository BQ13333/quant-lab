import pandas as pd

from quant_lab.validation.gates import GateDecision, ValidationGate


def test_promote_passes_with_strong_model():
    gate = ValidationGate()
    row = {
        "sharpe": 1.5,
        "deflated_sharpe": 0.98,
        "annual_excess_return": 0.05,
        "information_ratio": 0.6,
        "max_drawdown": -0.10,
        "turnover": 1.5,
        "cost_bps": 5.0,
        "returns": pd.Series([0.001] * 200),
    }
    decision = gate.promote(row)
    assert decision.passed is True
    assert decision.reasons == ()


def test_promote_fails_low_sharpe():
    gate = ValidationGate()
    row = {
        "sharpe": 0.5,
        "deflated_sharpe": 0.98,
        "annual_excess_return": 0.05,
        "information_ratio": 0.6,
        "max_drawdown": -0.10,
        "turnover": 1.5,
        "cost_bps": 5.0,
        "returns": pd.Series([0.001] * 200),
    }
    decision = gate.promote(row)
    assert decision.passed is False
    assert any("sharpe" in r for r in decision.reasons)


def test_promote_fails_low_dsr():
    gate = ValidationGate()
    row = {
        "sharpe": 1.3,
        "deflated_sharpe": 0.70,
        "annual_excess_return": 0.05,
        "information_ratio": 0.6,
        "max_drawdown": -0.10,
        "turnover": 1.5,
        "cost_bps": 5.0,
        "returns": pd.Series([0.001] * 200),
    }
    decision = gate.promote(row)
    assert decision.passed is False
    assert any("deflated" in r for r in decision.reasons)


def test_promote_fails_negative_excess_return():
    gate = ValidationGate()
    row = {
        "sharpe": 1.3,
        "deflated_sharpe": 0.98,
        "annual_excess_return": -0.01,
        "information_ratio": 0.6,
        "max_drawdown": -0.10,
        "turnover": 1.5,
        "cost_bps": 5.0,
        "returns": pd.Series([0.001] * 200),
    }
    decision = gate.promote(row)
    assert decision.passed is False
    assert any("excess" in r for r in decision.reasons)


def test_promote_fails_negative_ir():
    gate = ValidationGate()
    row = {
        "sharpe": 1.3,
        "deflated_sharpe": 0.98,
        "annual_excess_return": 0.05,
        "information_ratio": -0.2,
        "max_drawdown": -0.10,
        "turnover": 1.5,
        "cost_bps": 5.0,
        "returns": pd.Series([0.001] * 200),
    }
    decision = gate.promote(row)
    assert decision.passed is False
    assert any("information ratio" in r for r in decision.reasons)


def test_promote_fails_high_drawdown():
    gate = ValidationGate()
    row = {
        "sharpe": 1.3,
        "deflated_sharpe": 0.98,
        "annual_excess_return": 0.05,
        "information_ratio": 0.6,
        "max_drawdown": -0.25,
        "turnover": 1.5,
        "cost_bps": 5.0,
        "returns": pd.Series([0.001] * 200),
    }
    decision = gate.promote(row)
    assert decision.passed is False
    assert any("drawdown" in r for r in decision.reasons)


def test_promote_fails_high_turnover():
    gate = ValidationGate()
    row = {
        "sharpe": 1.3,
        "deflated_sharpe": 0.98,
        "annual_excess_return": 0.05,
        "information_ratio": 0.6,
        "max_drawdown": -0.10,
        "turnover": 5.0,
        "cost_bps": 5.0,
        "returns": pd.Series([0.001] * 200),
    }
    decision = gate.promote(row)
    assert decision.passed is False
    assert any("turnover" in r for r in decision.reasons)


def test_promote_survives_2x_cost():
    gate = ValidationGate()
    # Very high turnover -> 2x cost wipes out Sharpe
    returns = pd.Series([0.001] * 200)
    # Varying weights to produce actual turnover
    weights = pd.DataFrame({
        "a": [0.5, -0.5] * 100,
        "b": [0.5, 0.5] * 100,
    })
    row = {
        "sharpe": 1.3,
        "deflated_sharpe": 0.98,
        "annual_excess_return": 0.05,
        "information_ratio": 0.6,
        "max_drawdown": -0.10,
        "turnover": 10.0,
        "cost_bps": 50.0,
        "returns": returns,
    }
    decision = gate.promote(row, weights=weights)
    assert decision.passed is False
    assert any("cost" in r.lower() for r in decision.reasons)


def test_evaluate_backward_compatible():
    gate = ValidationGate(min_sharpe=0.5, max_drawdown_limit=-0.2, max_turnover=3.0, min_deflated_sharpe_probability=0.8)
    decision = gate.evaluate(sharpe=0.6, max_drawdown=-0.15, turnover=2.0, deflated_sharpe_probability=0.85)
    assert decision.passed is True
