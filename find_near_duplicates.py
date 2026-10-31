"""Find likely near-duplicate Sorghum photos for manual grouping before splitting.

This is a review aid, not an automatic label checker. It does not change images
or the existing manifest. It groups photos whose 64-bit difference hashes are
within the chosen Hamming distance; visually similar but non-duplicate photos
may also be included.

Run:
  python find_near_duplicates.py --zip "C:/Users/USER/Downloads/Sorghum Disease Image Dataset.zip" --manifest "E:/ASA model/sorghum_audit/manifest.csv" --out-dir "E:/ASA model/near_duplicate_review"
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def dhash(image_bytes: bytes) -> int:
    with Image.open(io.BytesIO(image_bytes)) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path, dest="zip_path")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--distance", type=int, default=4,
                        help="Maximum 64-bit dHash Hamming distance; default 4 (conservative)")
    args = parser.parse_args()
    if not args.zip_path.is_file() or not args.manifest.is_file():
        parser.error("ZIP or manifest path does not exist.")

    with args.manifest.open(newline="", encoding="utf-8-sig") as file_obj:
        rows = list(csv.DictReader(file_obj))
    rows = [r for r in rows if r.get("archive_path") and r.get("class_name")]
    for row in rows:
        row["archive_path"] = row["archive_path"].replace("\\", "/")

    hashes = []
    with zipfile.ZipFile(args.zip_path) as archive:
        archive_names = set(archive.namelist())
        for i, row in enumerate(rows, 1):
            if row["archive_path"] not in archive_names:
                raise SystemExit(f"Image path missing from ZIP: {row['archive_path']}")
            hashes.append(dhash(archive.read(row["archive_path"])))
            if i % 500 == 0 or i == len(rows):
                print(f"Hashed {i}/{len(rows)} images", flush=True)

    uf = UnionFind(len(rows))
    pairs = []
    # Exact and visually similar candidates, including across class folders.
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance <= args.distance:
                uf.union(i, j)
                pairs.append((i, j, distance))
        if (i + 1) % 500 == 0 or i + 1 == len(rows):
            print(f"Compared {i + 1}/{len(rows)} images", flush=True)

    components = defaultdict(list)
    for i in range(len(rows)):
        components[uf.find(i)].append(i)
    groups = [members for members in components.values() if len(members) > 1]
    groups.sort(key=lambda members: (-len(members), rows[min(members)]["archive_path"].casefold()))
    group_for = {}
    for group_number, members in enumerate(groups, 1):
        for i in members:
            group_for[i] = f"G{group_number:04d}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    group_path = args.out_dir / "candidate_groups.csv"
    with group_path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["candidate_group", "archive_path", "class_name", "old_split"])
        writer.writeheader()
        for members in groups:
            group_id = group_for[members[0]]
            for i in members:
                writer.writerow({"candidate_group": group_id, "archive_path": rows[i]["archive_path"],
                                 "class_name": rows[i]["class_name"], "old_split": rows[i].get("split", "")})

    pair_path = args.out_dir / "similar_pairs.csv"
    with pair_path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["candidate_group", "dhash_distance", "same_class", "image_1", "class_1", "image_2", "class_2"])
        for i, j, distance in sorted(pairs, key=lambda p: (p[2], rows[p[0]]["archive_path"].casefold())):
            writer.writerow([group_for[i], distance, rows[i]["class_name"] == rows[j]["class_name"],
                             rows[i]["archive_path"], rows[i]["class_name"],
                             rows[j]["archive_path"], rows[j]["class_name"]])

    print(f"Candidate groups (2+ images): {len(groups)}")
    print(f"Similar image pairs: {len(pairs)}")
    print(f"Review files: {group_path} and {pair_path}")
    print("Review candidates manually before using groups for a new split; visual similarity is not proof of same plant/session or wrong label.")


if __name__ == "__main__":
    main()
