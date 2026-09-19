"""Validate bundle hashes and file formats without rerunning scientific analyses."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    a=ap.parse_args()
    root=a.root.resolve()
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    errors=[];records=0
    for item in manifest["files"]:
        p=(root/item["path"]).resolve()
        if not p.is_relative_to(root):
            errors.append("Invalid manifest path");continue
        if not p.is_file():
            errors.append("Missing "+item["path"]);continue
        raw=p.read_bytes()
        if len(raw)!=item["bytes"] or hashlib.sha256(raw).hexdigest()!=item["sha256"]:
            errors.append("Checksum/size mismatch "+item["path"]);continue
        try:
            if p.suffix==".py":
                ast.parse(raw.decode("utf-8-sig"),filename=item["path"])
            elif p.suffix==".json":
                json.loads(raw.decode("utf-8-sig"));records+=1
            elif p.suffix==".jsonl":
                for line in raw.decode("utf-8-sig").splitlines():
                    if line.strip():
                        json.loads(line);records+=1
        except (SyntaxError,ValueError,UnicodeError) as e:
            errors.append(f"{item['path']}: {e}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(manifest['files'])} listed files and parsed {records} JSON/JSONL records.")
    print("Integrity and format check only; measurements were not recomputed.")

if __name__=="__main__":
    main()

