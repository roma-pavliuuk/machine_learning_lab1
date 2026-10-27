import yaml

from src.dynamic_pipeline_builder import DynamicPipelineBuilder


def test_idempotent(config_path):
    a = DynamicPipelineBuilder(config_path).build()
    b = DynamicPipelineBuilder(config_path).build()
    assert repr(a) == repr(b), "Збирання не ідемпотентне"


def test_protected_excluded(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ct = DynamicPipelineBuilder(config_path).build()
    used = {col for _, _, cols in ct.transformers for col in cols}
    leaked = used & set(cfg.get("protected_attributes", []))
    assert not leaked, f"Захищені ознаки в конвеєрі: {leaked}"


if __name__ == "__main__":
    for path in ("config/config_lr.yaml", "config/config_rf.yaml"):
        test_idempotent(path)
        test_protected_excluded(path)
        print(f"{path}: перевірки пройдено")
