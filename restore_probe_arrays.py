"""Restore archived probe arrays from JSON; no sampling or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent

def restore(source, out):
    obj = json.loads(source.read_text(encoding="utf-8"))
    if obj.get("format") != "numpy-arrays-json-v1":
        raise ValueError(f"Unsupported array format: {source}")
    rel = Path(obj["original_relative_path"])
    target = (out / rel).resolve()
    if not target.is_relative_to(out.resolve()):
        raise ValueError("Array output path escapes the destination")
    arrays = {}
    for key, item in obj["arrays"].items():
        arr = np.asarray(item["data"], dtype=np.dtype(item["dtype"])).reshape(item["shape"])
        got = hashlib.sha256(arr.tobytes(order="C")).hexdigest()
        if got != item["array_sha256"]:
            raise ValueError(f"Array checksum mismatch: {source.name}:{key}")
        arrays[key] = arr
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Choose a fresh output directory: {target}")
    np.savez_compressed(target, **arrays)
    print(f"Restored {rel.as_posix()} ({len(arrays)} arrays)")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "generated_inputs")
    a = ap.parse_args()
    for folder in ("ft_data", "nl_data"):
        for source in sorted((ROOT / folder).glob("*.arrays.json")):
            restore(source, a.out)
        for name in ("meta.json", "vocab.json"):
            source = ROOT / folder / name
            if source.exists():
                target = a.out / folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != source.read_bytes():
                    raise FileExistsError(f"Metadata already differs: {target}")
                shutil.copyfile(source, target)

if __name__ == "__main__":
    main()

