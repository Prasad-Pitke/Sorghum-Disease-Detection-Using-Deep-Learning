"""Audit a class-folder Sorghum image ZIP and create reproducible split manifests.

Usage:
  python -m pip install pillow
  python audit_sorghum.py "C:/path/Sorghum Disease Image Dataset.zip" --out-dir sorghum_audit

The script does not extract or modify the source archive. Exact duplicate files
are assigned to the same split within their class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    raise SystemExit("Pillow is required. Install it with: python -m pip install pillow")

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
SPLIT_NAMES = ("train", "validation", "test")


def sha256_stream(stream: object) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):  # type: ignore[attr-defined]
        digest.update(chunk)
    return digest.hexdigest()


def assign_group_splits(records: list[dict], seed: int) -> None:
    """Stratify by class while keeping identical-content groups together."""
    by_class: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        by_class[row["class_name"]][row["sha256"]].append(row)

    rng = random.Random(seed)
    for groups in by_class.values():
        group_items = list(groups.items())
        rng.shuffle(group_items)
        # Largest groups first gives a closer fit to 70/15/15 targets.
        group_items.sort(key=lambda item: len(item[1]), reverse=True)
        total = sum(len(items) for _, items in group_items)
        targets = {"train": total * 0.70, "validation": total * 0.15, "test": total * 0.15}
        counts = Counter()
        for _, items in group_items:
            split = min(SPLIT_NAMES, key=lambda name: counts[name] / targets[name])
            for row in items:
                row["split"] = split
            counts[split] += len(items)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path, help="Path to the dataset ZIP")
    parser.add_argument("--out-dir", type=Path, default=Path("sorghum_audit"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.zip_path.is_file():
        parser.error(f"ZIP file not found: {args.zip_path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    errors: list[dict] = []
    non_images: list[str] = []

    try:
        archive = zipfile.ZipFile(args.zip_path)
    except zipfile.BadZipFile:
        print("Error: the input file is not a readable ZIP archive.", file=sys.stderr)
        return 2

    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = Path(info.filename)
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                non_images.append(info.filename)
                continue
            parts = info.filename.replace("\\", "/").strip("/").split("/")
            if len(parts) < 2:
                errors.append({"archive_path": info.filename, "error": "No class folder"})
                continue
            class_name = parts[-2]
            try:
                with archive.open(info) as file_obj:
                    digest = sha256_stream(file_obj)
                # Reopen to let Pillow validate/decode image contents.
                with archive.open(info) as file_obj, Image.open(file_obj) as image:
                    image.verify()
                with archive.open(info) as file_obj, Image.open(file_obj) as image:
                    width, height = image.size
                records.append({
                    "archive_path": info.filename,
                    "class_name": class_name,
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "split": "",
                })
            except (UnidentifiedImageError, OSError, ValueError, zipfile.BadZipFile) as exc:
                errors.append({"archive_path": info.filename, "error": str(exc)})

    assign_group_splits(records, args.seed)

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=[
            "archive_path", "class_name", "sha256", "width", "height", "split"
        ])
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: row["archive_path"].casefold()))

    error_path = args.out_dir / "image_errors.csv"
    with error_path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["archive_path", "error"])
        writer.writeheader()
        writer.writerows(errors)

    counts = Counter(row["class_name"] for row in records)
    duplicate_groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        duplicate_groups[row["sha256"]].append(row)
    duplicate_rows = [items for items in duplicate_groups.values() if len(items) > 1]

    print(f"Readable JPEG images: {len(records)}")
    print(f"Unreadable images: {len(errors)} (see {error_path})")
    print(f"Non-JPEG files ignored: {len(non_images)}")
    print("Class counts:")
    for class_name, count in sorted(counts.items(), key=lambda item: item[0].casefold()):
        split_counts = Counter(row["split"] for row in records if row["class_name"] == class_name)
        print(f"  {class_name}: {count} | train {split_counts['train']}, "
              f"validation {split_counts['validation']}, test {split_counts['test']}")
    print(f"Exact duplicate groups: {len(duplicate_rows)}")
    print(f"Manifest saved to: {manifest_path.resolve()}")
    print("Note: exact hashes do not detect resized, cropped, or near-identical photos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
