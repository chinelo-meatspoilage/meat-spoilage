"""Rebalance training set by moving a deterministic subset of HALF-FRESH images
out of train/spoiled/ into train/_excluded_half_spoiled/.

Why: the Spoiled class became inflated (1154 vs 762 Fresh) because half-fresh
images were merged into Spoiled. Moving 392 of them out makes the classes 762:762
without losing any Fresh data and without deleting anything from disk.

Reversal: the manifest at train/_excluded_half_spoiled/_manifest.txt lists every
moved file. To revert, move them back into train/spoiled/ and re-add the removed
CSV rows from _classes.csv.bak.
"""

from __future__ import annotations

import csv
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "project-dataset" / "train"
CSV_PATH = ROOT / "_classes.csv"
SPOILED_DIR = ROOT / "spoiled"
EXCLUDED_DIR = ROOT / "_excluded_half_spoiled"
MANIFEST = EXCLUDED_DIR / "_manifest.txt"
BACKUP_CSV = ROOT / "_classes.csv.bak"

NUM_TO_MOVE = 392
SEED = 42


def main(dry_run: bool) -> None:
    if not CSV_PATH.exists():
        sys.exit(f"CSV not found: {CSV_PATH}")
    if not SPOILED_DIR.exists():
        sys.exit(f"Spoiled folder not found: {SPOILED_DIR}")

    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header, data = rows[0], rows[1:]
    if header != ["filename", "Fresh", "Spoiled"]:
        sys.exit(f"Unexpected CSV header: {header}")

    half_fresh_rows = [r for r in data if r[0].startswith("HALF-FRESH-")]
    other_rows = [r for r in data if not r[0].startswith("HALF-FRESH-")]

    print(f"CSV totals before: Fresh+Spoiled rows={len(data)}, "
          f"HALF-FRESH rows={len(half_fresh_rows)}, other rows={len(other_rows)}")

    if len(half_fresh_rows) < NUM_TO_MOVE:
        sys.exit(f"Not enough HALF-FRESH rows ({len(half_fresh_rows)}) "
                 f"to move {NUM_TO_MOVE}")

    rng = random.Random(SEED)
    to_move = sorted(rng.sample(half_fresh_rows, NUM_TO_MOVE),
                     key=lambda r: r[0])

    # Verify every selected file exists on disk before moving anything.
    missing = [r[0] for r in to_move if not (SPOILED_DIR / r[0]).exists()]
    if missing:
        sys.exit(f"Aborting: {len(missing)} selected files are missing on disk. "
                 f"First 5: {missing[:5]}")

    keep_filenames = {r[0] for r in to_move}
    new_data = [r for r in data if r[0] not in keep_filenames]

    fresh_after = sum(1 for r in new_data if r[1] == "1" and r[2] == "0")
    spoiled_after = sum(1 for r in new_data if r[1] == "0" and r[2] == "1")

    print(f"\nPlanned result:")
    print(f"  Files to move:    {len(to_move)}")
    print(f"  CSV rows after:   {len(new_data)} "
          f"(Fresh={fresh_after}, Spoiled={spoiled_after})")
    print(f"  Destination:      {EXCLUDED_DIR}")
    print(f"  Backup CSV:       {BACKUP_CSV}")

    if dry_run:
        print("\n[DRY RUN] No files moved, no CSV written.")
        print(f"First 3 of {NUM_TO_MOVE} selected: "
              f"{[r[0] for r in to_move[:3]]}")
        return

    # Backup CSV before modifying.
    shutil.copy2(CSV_PATH, BACKUP_CSV)
    print(f"\nBacked up CSV -> {BACKUP_CSV}")

    EXCLUDED_DIR.mkdir(parents=True, exist_ok=True)

    moved = []
    for row in to_move:
        src = SPOILED_DIR / row[0]
        dst = EXCLUDED_DIR / row[0]
        shutil.move(str(src), str(dst))
        moved.append(row[0])

    MANIFEST.write_text("\n".join(moved) + "\n", encoding="utf-8")
    print(f"Moved {len(moved)} files. Manifest -> {MANIFEST}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(new_data)
    print(f"Rewrote CSV -> {CSV_PATH}")
    print(f"\nDone. Fresh={fresh_after}, Spoiled={spoiled_after} "
          f"(balanced 1:1).")


if __name__ == "__main__":
    main(dry_run="--apply" not in sys.argv)
