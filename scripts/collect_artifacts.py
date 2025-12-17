"""Move generated *.pkl artifacts into ./artifacts for a cleaner repo.

Run after executing the notebook.
"""
from __future__ import annotations
import os
import shutil

ARTIFACTS = [
    "dataset_final_fixed.pkl",
    "dataset_with_embeddings.pkl",
    "dataset_with_embeddings_fixed.pkl",
    "smart_adapter.pkl",
    "multilabel_binarizer.pkl",
    "scaler.pkl",
    "calibrators.pkl",
    "per_class_thresholds.pkl",
]

out_dir = "artifacts"
os.makedirs(out_dir, exist_ok=True)

moved = 0
for f in ARTIFACTS:
    if os.path.exists(f):
        shutil.move(f, os.path.join(out_dir, f))
        print(f"Moved: {f} -> {out_dir}/{f}")
        moved += 1

print(f"Done. Moved {moved} file(s).")
