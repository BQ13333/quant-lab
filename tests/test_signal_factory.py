from __future__ import annotations

import pandas as pd

from quant_lab.models import (
    SignalFactoryConfig,
    build_signal_models,
    default_signal_specs,
    signal_manifest,
)


def test_signal_factory_lite_profile_names_models():
    models = build_signal_models(SignalFactoryConfig(prefix="test", profile="lite", include_group_models=False))
    names = [model.name for model in models]

    assert names == [
        "test_mom_21",
        "test_mom_63",
        "test_vam_63",
        "test_rev_5",
        "test_lowvol_63",
    ]


def test_signal_factory_slow_profile_with_groups():
    groups = pd.Series({"A": "US", "B": "US", "C": "Europe"})
    models = build_signal_models(SignalFactoryConfig(prefix="slow", profile="slow", groups=groups))
    manifest = signal_manifest(models)

    assert len(models) >= 10
    assert "slow_group_rel_mom_126" in set(manifest["model_name"])
    assert {"model_name", "class_name", "lookback", "horizon"}.issubset(manifest.columns)


def test_default_signal_specs_documents_research_queue():
    specs = default_signal_specs()

    assert any(spec.family == "ensemble" for spec in specs)
    assert all(spec.horizon_days > 0 for spec in specs)
