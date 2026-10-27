import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from src.dynamic_pipeline_builder import DynamicPipelineBuilder

from conftest import CONFIGS, ROOT

SRC = ROOT / "src" / "dynamic_pipeline_builder.py"


def test_configs_differ_only_in_model_family():
    lr = CONFIGS[0].read_text(encoding="utf-8").splitlines()
    rf = CONFIGS[1].read_text(encoding="utf-8").splitlines()
    assert len(lr) == len(rf)
    diff = [(a, b) for a, b in zip(lr, rf) if a != b]
    assert len(diff) == 1 and all(line.startswith("model_family:") for line in diff[0])


def test_fit_on_train_transform_on_test(config_path, split):
    X_train, X_test, y_train, _ = split
    builder = DynamicPipelineBuilder(config_path)
    ct = builder.build()
    Z_train = ct.fit_transform(X_train, y_train)
    Z_test = ct.transform(X_test)
    assert Z_train.shape[1] == Z_test.shape[1]
    assert not np.isnan(Z_train).any() and not np.isnan(Z_test).any()
    names = ct.get_feature_names_out()
    for attr in builder.protected_attributes + [builder.target]:
        assert not any(name.split("__", 1)[1].startswith(attr) for name in names)


def test_audit_trail_covers_every_feature(config_path, split):
    X_train, _, y_train, _ = split
    builder = DynamicPipelineBuilder(config_path)
    ct = builder.build().fit(X_train, y_train)
    audit = DynamicPipelineBuilder.audit_trail(ct)
    expected = [c for cols in builder.config["features"].values() for c in cols]
    assert list(audit["feature"]) == expected
    assert set(zip(audit["strategy"], audit["add_indicator"])) == {
        ("median", False), ("median", True), ("constant", True),
        ("most_frequent", False), ("constant", False),
    }


def test_repeated_fit_is_deterministic(config_path, split):
    X_train, _, y_train, _ = split
    a = DynamicPipelineBuilder(config_path).build().fit_transform(X_train, y_train)
    b = DynamicPipelineBuilder(config_path).build().fit_transform(X_train, y_train)
    assert np.array_equal(a, b)


def _source_literals():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    strings, floats = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                strings.add(node.value)
            elif isinstance(node.value, float):
                floats.add(node.value)
    return strings, floats


def test_no_dataset_specific_literals_in_code():
    strings, floats = _source_literals()
    forbidden_strings, forbidden_floats = set(), set()
    for path in CONFIGS:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        forbidden_strings |= set(cfg["transform_policy"])
        forbidden_strings |= {c for cols in cfg["features"].values() for c in cols}
        forbidden_strings |= set(cfg["protected_attributes"]) | {cfg["target"]}
        forbidden_strings |= set(cfg["features"])
        for rules in cfg["imputation_policy"].values():
            forbidden_floats |= {float(r["max_share"]) for r in rules}
            forbidden_strings |= {r["fill_value"] for r in rules if isinstance(r.get("fill_value"), str)}
    assert not strings & forbidden_strings, strings & forbidden_strings
    assert not floats & forbidden_floats, floats & forbidden_floats


def test_no_statistics_outside_pipeline_in_code():
    source = SRC.read_text(encoding="utf-8")
    assert ".fillna(" not in source
    assert "read_csv" not in source


def test_run_verification_script_passes():
    result = subprocess.run(
        [sys.executable, "run_verification.py"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("перевірки пройдено") == 2
