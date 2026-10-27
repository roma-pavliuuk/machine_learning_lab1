import copy

import pytest
import yaml
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.dynamic_pipeline_builder import DynamicPipelineBuilder, validate_config

from conftest import CONFIGS, FIXTURES, write_config


def test_protected_leak_fixture_is_blocked():
    with pytest.raises(ValueError, match="Захищені ознаки потрапили у features"):
        DynamicPipelineBuilder(str(FIXTURES / "config_protected_leak.yaml"))


def test_validate_config_accepts_valid_configs(config_path):
    with open(config_path, encoding="utf-8") as f:
        validate_config(yaml.safe_load(f))


def test_build_returns_unfitted_transformer(config_path):
    ct = DynamicPipelineBuilder(config_path).build()
    with pytest.raises(NotFittedError):
        check_is_fitted(ct)


def test_build_respects_model_family(config_path, base_config):
    ct = DynamicPipelineBuilder(config_path).build()
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    policy = cfg["transform_policy"][cfg["model_family"]]
    for group, pipe, _ in ct.transformers:
        names = [name for name, _ in pipe.steps]
        expected = ["impute"] + [s if isinstance(s, str) else next(iter(s)) for s in policy[group]]
        assert names == expected


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.update(model_family="unknown_family"), "не описано в transform_policy"),
        (lambda c: c["features"]["numeric_continuous"].append(c["target"]), "прямий витік цілі"),
        (lambda c: c["features"]["numeric_discrete"].append("duration"), "входить до груп"),
        (lambda c: c["imputation_routing"].pop("numeric_discrete"), "не задано imputation_routing"),
        (lambda c: c["imputation_policy"]["numeric"].pop(), "не покриває частку 1"),
        (lambda c: c["imputation_policy"]["numeric"].reverse(), "мають зростати"),
        (lambda c: c["imputation_policy"]["numeric"][0].update(strategy="mean_of_test"), "невідома стратегія"),
        (lambda c: c["transform_policy"]["partitioning"]["numeric_continuous"].append("pca"), "невідомі методи"),
        (lambda c: c["transform_policy"]["partitioning"].pop("numeric_discrete"), "не описує групу"),
        (lambda c: c["ordinal_categories"].pop("job"), "ordinal_categories"),
        (lambda c: c.pop("protected_attributes"), "відсутні секції"),
    ],
    ids=[
        "unknown_model_family", "target_in_features", "column_in_two_groups",
        "missing_routing", "policy_not_covering_1", "unsorted_thresholds",
        "unknown_strategy", "unknown_method", "family_missing_group",
        "missing_ordinal_levels", "missing_section",
    ],
)
def test_schema_violations_fail_loud(tmp_path, base_config, mutate, message):
    cfg = copy.deepcopy(base_config)
    mutate(cfg)
    path = write_config(tmp_path, cfg)
    with pytest.raises(ValueError, match=message):
        DynamicPipelineBuilder(path)


def test_new_feature_needs_no_code_change(tmp_path, base_config):
    cfg = copy.deepcopy(base_config)
    cfg["features"]["numeric_discrete"].append("brand_new_column")
    ct = DynamicPipelineBuilder(write_config(tmp_path, cfg)).build()
    used = {col for _, _, cols in ct.transformers for col in cols}
    assert "brand_new_column" in used


def test_all_configs_exist():
    assert all(p.exists() for p in CONFIGS)
