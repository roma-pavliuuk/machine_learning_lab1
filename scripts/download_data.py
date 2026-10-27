import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

EXPECTED_SHA256 = {
    "german.data": "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871",
    "german.doc": "69b92cb551111bc37098698288849c96f0fd87ce81a2e54451ff39aa1e53fe49",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Завантаження {URL}")
    with urllib.request.urlopen(URL, timeout=60) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))

    for name, expected in EXPECTED_SHA256.items():
        content = archive.read(name)
        actual = sha256(content)
        if actual != expected:
            sys.exit(f"{name}: SHA-256 {actual} не збігається з очікуваним {expected}")
        (RAW_DIR / name).write_bytes(content)
        print(f"{name}: OK ({actual[:12]}…)")


if __name__ == "__main__":
    main()
