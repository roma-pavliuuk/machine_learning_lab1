from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted


def validate_config(config: Dict[str, Any]) -> None:
    protected = set(config.get("protected_attributes", []))
    features = set()
    for group in config.get("features", {}).values():
        features.update(group)
    leaked = protected & features
    if leaked:
        raise ValueError(f"Захищені ознаки потрапили у features: {leaked}")


_IMPUTE_STRATEGIES = ("median", "most_frequent", "constant")

_REQUIRED_SECTIONS = (
    "model_family",
    "features",
    "protected_attributes",
    "imputation_policy",
    "imputation_routing",
    "transform_policy",
)


def _parse_step(spec: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict) and len(spec) == 1:
        name, params = next(iter(spec.items()))
        return name, dict(params or {})
    raise ValueError(f"Некоректний опис кроку transform_policy: {spec!r}")


def validate_schema(config: Dict[str, Any]) -> None:
    missing = [s for s in _REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(f"У конфігурації відсутні секції: {missing}")

    families = config["transform_policy"]
    family = config["model_family"]
    if family not in families:
        raise ValueError(
            f"model_family={family!r} не описано в transform_policy "
            f"(доступні: {sorted(families)})"
        )

    features: Dict[str, List[str]] = config["features"]
    seen: Dict[str, str] = {}
    for group, columns in features.items():
        for col in columns or []:
            if col in seen:
                raise ValueError(f"Колонка {col!r} входить до груп {seen[col]!r} і {group!r}")
            seen[col] = group

    target = config.get("target")
    if target is not None and target in seen:
        raise ValueError(f"Цільова змінна {target!r} внесена до features — прямий витік цілі")
    if target is not None and target in set(config["protected_attributes"]):
        raise ValueError(f"Цільова змінна {target!r} не може бути захищеною ознакою")

    policies = config["imputation_policy"]
    routing = config["imputation_routing"]
    for group in features:
        if group not in routing:
            raise ValueError(f"Для групи {group!r} не задано imputation_routing")
        if routing[group] not in policies:
            raise ValueError(
                f"Група {group!r} посилається на невідому політику імпутації {routing[group]!r}"
            )

    for name, rules in policies.items():
        _validate_imputation_rules(name, rules)

    ordinal_levels = config.get("ordinal_categories", {})
    for fam, groups in families.items():
        for group, columns in features.items():
            if group not in groups:
                raise ValueError(f"transform_policy.{fam} не описує групу {group!r}")
            names = [_parse_step(s)[0] for s in groups[group] or []]
            unknown = [n for n in names if n not in _STEP_FACTORIES]
            if unknown:
                raise ValueError(f"transform_policy.{fam}.{group}: невідомі методи {unknown}")
            if len(set(names)) != len(names):
                raise ValueError(f"transform_policy.{fam}.{group}: кроки повторюються {names}")
            if _ORDINAL_STEP in names:
                absent = [c for c in columns or [] if c not in ordinal_levels]
                if absent:
                    raise ValueError(f"Для {absent} не задано рівні в ordinal_categories")
                rules = policies[routing[group]]
                if any(r.get("add_indicator") for r in rules):
                    raise ValueError(
                        f"Група {group!r}: порядковий кодувальник несумісний з "
                        f"індикатором пропуску в політиці {routing[group]!r}"
                    )


def _validate_imputation_rules(name: str, rules: List[Dict[str, Any]]) -> None:
    if not rules:
        raise ValueError(f"Політика імпутації {name!r} порожня")
    previous = None
    for rule in rules:
        if "max_share" not in rule or "strategy" not in rule:
            raise ValueError(f"{name}: правило {rule} має містити max_share і strategy")
        if rule["strategy"] not in _IMPUTE_STRATEGIES:
            raise ValueError(f"{name}: невідома стратегія {rule['strategy']!r}")
        if rule["strategy"] == "constant" and "fill_value" not in rule:
            raise ValueError(f"{name}: стратегія constant потребує fill_value ({rule})")
        if previous is not None and rule["max_share"] <= previous:
            raise ValueError(f"{name}: пороги max_share мають зростати ({rule})")
        previous = rule["max_share"]
    if previous < 1:
        raise ValueError(f"{name}: остання межа max_share={previous} не покриває частку 1")


class AdaptiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self, policy: List[Dict]):
        self.policy = policy

    def fit(self, X, y=None):
        X = _as_frame(X)
        rules = sorted(self.policy, key=lambda r: r["max_share"])

        shares = X.isna().to_numpy().mean(axis=0)

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.fill_values_ = {}
        self.audit_ = {}
        for col, share in zip(X.columns, shares):
            rule = _match_rule(col, float(share), rules)
            fill = _fill_value(col, X[col], rule)
            self.fill_values_[col] = fill
            self.audit_[col] = {
                "missing_share": float(share),
                "rule": dict(rule),
                "fill_value": fill,
                "add_indicator": bool(rule.get("add_indicator", False)),
            }
        return self

    def transform(self, X):
        check_is_fitted(self, "audit_")
        X = _as_frame(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                f"Колонки на вході transform {list(X.columns)} не збігаються з "
                f"колонками під час fit {list(self.feature_names_in_)}"
            )

        filled = {}
        indicators = {}
        for col in X.columns:
            mask = X[col].isna()
            if self.audit_[col]["add_indicator"]:
                indicators[_indicator_name(col)] = mask.astype(float)
            filled[col] = X[col].astype(object).where(~mask, self.fill_values_[col])

        out = pd.DataFrame(filled, index=X.index)
        out = out.infer_objects()
        if indicators:
            out = pd.concat([out, pd.DataFrame(indicators, index=X.index)], axis=1)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "audit_")
        names = list(self.feature_names_in_)
        names += [_indicator_name(c) for c in self.feature_names_in_ if self.audit_[c]["add_indicator"]]
        return np.asarray(names, dtype=object)


def _as_frame(X) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _indicator_name(col) -> str:
    return f"{col}_missing"


def _match_rule(col, share: float, rules: List[Dict]) -> Dict:
    for rule in rules:
        if share <= rule["max_share"]:
            return rule
    raise ValueError(f"Для колонки {col!r} з часткою пропусків {share:.3f} немає правила")


def _fill_value(col, values: pd.Series, rule: Dict):
    strategy = rule["strategy"]
    if strategy == "constant":
        return rule["fill_value"]
    observed = values.dropna()
    if observed.empty:
        raise ValueError(
            f"Колонка {col!r} не має спостережених значень — стратегія {strategy!r} "
            f"не визначена; політика має маршрутизувати її до constant"
        )
    if strategy == "median":
        return float(np.median(pd.to_numeric(observed).to_numpy(dtype=float)))
    if strategy == "most_frequent":
        return observed.mode().iloc[0]
    raise ValueError(f"Невідома стратегія {strategy!r}")


class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = _as_frame(X)
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("Потрібно 0 <= lower_quantile < upper_quantile <= 1")
        values = X.to_numpy(dtype=float)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.lower_ = np.nanquantile(values, self.lower_quantile, axis=0)
        self.upper_ = np.nanquantile(values, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        values = _as_frame(X).to_numpy(dtype=float)
        return np.clip(values, self.lower_, self.upper_)


class WoEEncoder(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("WoEEncoder потребує цільову змінну y у fit")
        X = _as_frame(X)
        y = np.asarray(y)
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("WoEEncoder очікує бінарну ціль зі значеннями 0/1")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.mapping_ = {}
        self.iv_ = {}
        a = self.smoothing
        for col in X.columns:
            table = pd.crosstab(X[col].to_numpy(), y)
            table = table.reindex(columns=[0, 1], fill_value=0)
            k = len(table)
            p_good = (table[0] + a) / (table[0].sum() + a * k)
            p_bad = (table[1] + a) / (table[1].sum() + a * k)
            woe = np.log(p_good / p_bad)
            self.mapping_[col] = woe.to_dict()
            self.iv_[col] = float(((p_good - p_bad) * woe).sum())
        return self

    def transform(self, X):
        check_is_fitted(self, "mapping_")
        X = _as_frame(X)
        out = np.empty(X.shape, dtype=float)
        for j, col in enumerate(self.feature_names_in_):
            mapping = self.mapping_[col]
            out[:, j] = [mapping.get(v, 0.0) for v in X.iloc[:, j]]
        return out


_ORDINAL_STEP = "ordinal_encoder"
_STEP_FACTORIES = {
    "log1p": lambda **p: FunctionTransformer(np.log1p, feature_names_out="one-to-one", **p),
    "standard_scaler": lambda **p: StandardScaler(**p),
    "winsorize": lambda **p: Winsorizer(**p),
    "woe": lambda **p: WoEEncoder(**p),
    "one_hot": lambda **p: OneHotEncoder(handle_unknown="ignore", sparse_output=False, **p),
    _ORDINAL_STEP: None,
}


class DynamicPipelineBuilder:
    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        validate_config(self.config)
        validate_schema(self.config)

    @property
    def protected_attributes(self) -> List[str]:
        return list(self.config.get("protected_attributes", []))

    @property
    def target(self):
        return self.config.get("target")

    def build(self) -> ColumnTransformer:
        model_family = self.config["model_family"]
        policy = self.config["transform_policy"][model_family]

        transformers = []
        for group, columns in self.config["features"].items():
            if not columns:
                continue
            steps = [("impute", AdaptiveImputer(policy=self._imputation_rules(group)))]
            for spec in policy[group] or []:
                name, params = _parse_step(spec)
                steps.append((name, self._make_step(name, params, group, columns)))
            transformers.append((group, Pipeline(steps), list(columns)))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _imputation_rules(self, group: str) -> List[Dict]:
        name = self.config["imputation_routing"][group]
        return copy.deepcopy(self.config["imputation_policy"][name])

    def _make_step(self, name: str, params: Dict[str, Any], group: str, columns: List[str]):
        if name == _ORDINAL_STEP:
            levels = [self._ordinal_levels(group, col) for col in columns]
            return OrdinalEncoder(
                categories=levels,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                **params,
            )
        return _STEP_FACTORIES[name](**params)

    def _ordinal_levels(self, group: str, column: str) -> List:
        levels = list(self.config["ordinal_categories"][column])
        for rule in self._imputation_rules(group):
            if rule["strategy"] == "constant" and rule["fill_value"] not in levels:
                levels.append(rule["fill_value"])
        return levels

    @staticmethod
    def audit_trail(fitted: ColumnTransformer) -> pd.DataFrame:
        check_is_fitted(fitted)
        rows = []
        for group, pipe, _ in fitted.transformers_:
            if not isinstance(pipe, Pipeline):
                continue
            imputer = pipe.steps[0][1]
            downstream = [name for name, _ in pipe.steps[1:]]
            for col, entry in imputer.audit_.items():
                rule = entry["rule"]
                rows.append({
                    "feature": col,
                    "group": group,
                    "missing_share": round(entry["missing_share"], 4),
                    "max_share": rule["max_share"],
                    "strategy": rule["strategy"],
                    "fill_value": entry["fill_value"],
                    "add_indicator": entry["add_indicator"],
                    "transforms": " → ".join(downstream) or "(лише імпутація)",
                })
        return pd.DataFrame(rows)
