"""One-shot: convert train/spoiled/new sm.avif to JPEG and update _classes.csv.

Reason: TensorFlow's tf.io.decode_image cannot read AVIF, so phase2_training.py
crashes on this single file. We composite onto white (in case of alpha), save
as .jpg, delete the original, and rewrite the CSV row.
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

try:
    import pillow_avif  # noqa: F401  enables AVIF in Pillow
except ImportError:
    raise SystemExit("pillow-avif-plugin is required. Install: "
                     "pip install pillow-avif-plugin")

ROOT = Path(__file__).resolve().parent / "project-dataset" / "train"
SRC = ROOT / "spoiled" / "new sm.avif"
DST = ROOT / "spoiled" / "new sm.jpg"
CSV_PATH = ROOT / "_classes.csv"

if not SRC.exists():
    raise SystemExit(f"Source not found: {SRC}")
if DST.exists():
    raise SystemExit(f"Destination already exists, refusing to overwrite: {DST}")

img = Image.open(SRC)
if img.mode in ("RGBA", "P", "PA", "LA"):
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[3])
    img = bg
else:
    img = img.convert("RGB")
img.save(DST, "JPEG", quality=95)
print(f"Converted: {SRC.name} -> {DST.name}")

# Update CSV: rename the row's filename, keep label.
with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))

updated = 0
for r in rows[1:]:
    if r[0] == "new sm.avif":
        r[0] = "new sm.jpg"
        updated += 1
if updated != 1:
    raise SystemExit(f"Expected exactly 1 CSV row to update, got {updated}")

with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
print(f"Updated CSV row: 'new sm.avif' -> 'new sm.jpg'")

SRC.unlink()
print(f"Deleted: {SRC}")
print("Done.")
