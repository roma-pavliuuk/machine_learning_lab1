import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "german.data"
OUT_DIR = ROOT / "data" / "processed"

SEED = 42

COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose", "credit_amount",
    "savings_account", "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "num_dependents", "telephone",
    "foreign_worker", "risk",
]


CODEBOOK = {
    "checking_account": {"A11": "<0 DM", "A12": "0-200 DM", "A13": ">=200 DM", "A14": "no account"},
    "credit_history": {
        "A30": "no credits / all paid", "A31": "all paid at this bank",
        "A32": "existing paid duly", "A33": "past delays", "A34": "critical account",
    },
    "purpose": {
        "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
        "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs",
        "A46": "education", "A47": "vacation", "A48": "retraining",
        "A49": "business", "A410": "others",
    },
    "savings_account": {
        "A61": "<100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
        "A64": ">=1000 DM", "A65": "unknown/none",
    },
    "employment_since": {
        "A71": "unemployed", "A72": "<1 year", "A73": "1-4 years",
        "A74": "4-7 years", "A75": ">=7 years",
    },
    "personal_status_sex": {
        "A91": "male: divorced/separated", "A92": "female: divorced/separated/married",
        "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single",
    },
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {
        "A121": "real estate", "A122": "savings agreement/life insurance",
        "A123": "car or other", "A124": "unknown/none",
    },
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemployed/unskilled non-resident", "A172": "unskilled resident",
        "A173": "skilled employee", "A174": "management/highly qualified",
    },
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

SEX_FROM_CODE = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}


INJECTION_PLAN = {
    "age": 0.05,
    "credit_amount": 0.20,
    "residence_since": 0.50,
    "savings_account": 0.06,
    "job": 0.03,
    "employment_since": 0.15,
    "purpose": 0.25,
}


def load_decoded() -> pd.DataFrame:
    raw = pd.read_csv(RAW, sep=" ", header=None, names=COLUMNS)
    df = raw.copy()
    df["sex"] = raw["personal_status_sex"].map(SEX_FROM_CODE)
    for col, mapping in CODEBOOK.items():
        df[col] = raw[col].map(mapping)
    df["bad_credit"] = (raw["risk"] == 2).astype(int)
    df = df.drop(columns="risk")
    unmapped = df[list(CODEBOOK)].isna().sum()
    if unmapped.any():
        raise ValueError(f"Невідомі коди в сирих даних: {unmapped[unmapped > 0].to_dict()}")
    return df


def inject_mcar(df: pd.DataFrame, plan: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    for col, share in plan.items():
        idx = rng.choice(n, size=round(share * n), replace=False)
        out.loc[out.index[idx], col] = np.nan
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = load_decoded()
    mcar = inject_mcar(clean, INJECTION_PLAN, SEED)

    clean_path = OUT_DIR / "german_credit_clean.csv"
    mcar_path = OUT_DIR / "german_credit_mcar.csv"
    clean.to_csv(clean_path, index=False)
    mcar.to_csv(mcar_path, index=False)

    manifest = {
        "source": "Statlog (German Credit Data), UCI ML Repository, dataset 144",
        "raw_sha256": sha256_file(RAW),
        "mechanism": "MCAR",
        "seed": SEED,
        "rng": "numpy.random.default_rng",
        "plan": INJECTION_PLAN,
        "n_rows": len(mcar),
        "missing_counts": {c: int(mcar[c].isna().sum()) for c in INJECTION_PLAN},
        "outputs": {
            clean_path.name: sha256_file(clean_path),
            mcar_path.name: sha256_file(mcar_path),
        },
    }
    (OUT_DIR / "injection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
