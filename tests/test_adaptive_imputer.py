import numpy as np
import pandas as pd
import pytest

from src.dynamic_pipeline_builder import AdaptiveImputer

POLICY = [
    {"max_share": 0.10, "strategy": "median"},
    {"max_share": 0.40, "strategy": "median", "add_indicator": True},
    {"max_share": 1.00, "strategy": "constant", "fill_value": 0, "add_indicator": True},
]


def frame_with_shares(shares, n=100):
    data = {}
    for name, share in shares.items():
        col = np.arange(1, n + 1, dtype=float)
        col[: round(share * n)] = np.nan
        data[name] = col
    return pd.DataFrame(data)


def test_routing_by_missing_share():
    X = frame_with_shares({"a": 0.0, "b": 0.10, "c": 0.11, "d": 0.40, "e": 0.41, "f": 1.0})
    imp = AdaptiveImputer(POLICY).fit(X)
    chosen = {c: imp.audit_[c]["rule"]["max_share"] for c in X.columns}
    assert chosen == {"a": 0.10, "b": 0.10, "c": 0.40, "d": 0.40, "e": 1.00, "f": 1.00}
    assert imp.audit_["c"]["add_indicator"] and not imp.audit_["b"]["add_indicator"]


def test_audit_trail_contains_share_and_rule():
    X = frame_with_shares({"a": 0.2})
    imp = AdaptiveImputer(POLICY).fit(X)
    entry = imp.audit_["a"]
    assert entry["missing_share"] == pytest.approx(0.2)
    assert entry["rule"]["strategy"] == "median"
    assert entry["fill_value"] == pytest.approx(np.median(np.arange(21, 101)))


def test_transform_does_not_recompute_state():
    train = frame_with_shares({"a": 0.2})
    test = frame_with_shares({"a": 0.5})
    imp = AdaptiveImputer(POLICY).fit(train)
    before = {k: dict(v) for k, v in imp.audit_.items()}
    out = imp.transform(test)
    assert imp.audit_ == before
    assert (out.loc[:49, "a"] == imp.fill_values_["a"]).all()
    assert list(out.columns) == ["a", "a_missing"]


def test_indicator_dimension_is_fixed_at_fit():
    train = frame_with_shares({"a": 0.0, "b": 0.3})
    test = frame_with_shares({"a": 0.3, "b": 0.0})
    imp = AdaptiveImputer(POLICY).fit(train)
    assert imp.transform(train).shape == imp.transform(test).shape
    assert not imp.transform(test).isna().any().any()


def test_target_is_ignored():
    X = frame_with_shares({"a": 0.2, "b": 0.5})
    y = np.r_[np.zeros(50), np.ones(50)]
    with_y = AdaptiveImputer(POLICY).fit(X, y)
    without_y = AdaptiveImputer(POLICY).fit(X)
    assert with_y.audit_ == without_y.audit_


def test_categorical_policy():
    policy = [
        {"max_share": 0.10, "strategy": "most_frequent"},
        {"max_share": 1.00, "strategy": "constant", "fill_value": "Missing"},
    ]
    X = pd.DataFrame({
        "low": ["x", "x", "y", None] + ["y"] * 46,
        "high": [None] * 20 + ["u", "v"] * 15,
    })
    out = AdaptiveImputer(policy).fit(X).transform(X)
    assert out.loc[3, "low"] == "y"
    assert (out.loc[:19, "high"] == "Missing").all()


def test_median_on_all_missing_fails_loud():
    policy = [{"max_share": 1.00, "strategy": "median"}]
    with pytest.raises(ValueError, match="не має спостережених значень"):
        AdaptiveImputer(policy).fit(pd.DataFrame({"a": [np.nan, np.nan]}))
