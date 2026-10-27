import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

CONFIGS = [ROOT / "config" / "config_lr.yaml", ROOT / "config" / "config_rf.yaml"]
FIXTURES = TESTS / "fixtures"
DATA = ROOT / "data" / "processed" / "german_credit_mcar.csv"


def write_config(tmp_path, cfg) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(path)


@pytest.fixture(params=CONFIGS, ids=lambda p: p.name)
def config_path(request) -> str:
    return str(request.param)


@pytest.fixture
def base_config() -> dict:
    with open(CONFIGS[0], encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def split():
    df = pd.read_csv(DATA)
    y = df.pop("bad_credit")
    return train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)
