"""Export a trained model's misclassified validation or test images for review.

Run:
  python inspect_sorghum_errors.py --zip "C:/Users/USER/Downloads/Sorghum Disease Image Dataset.zip" --manifest "E:/ASA model/sorghum_audit/manifest.csv" --model "E:/ASA model/results/sorghum_cnn_best.pt" --out-dir "E:/ASA model/results/error_review"
  Add --split validation to review validation errors when planning model changes.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_sorghum import SorghumCNN


class TestImages(Dataset):
    def __init__(self, rows: list[dict], class_to_id: dict[str, int], size: int, zip_path: Path):
        self.rows = rows
        self.class_to_id = class_to_id
        self.size = size
        self.zip_path = zip_path
        self._zip = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        raw = self._zip.read(row["archive_path"])
        with Image.open(io.BytesIO(raw)) as im:
            original = im.convert("RGB")
            resized = original.copy()
            resized.thumbnail((self.size, self.size), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
            canvas.paste(resized, ((self.size - resized.width) // 2, (self.size - resized.height) // 2))
            arr = np.asarray(canvas, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(((arr - 0.5) / 0.5).transpose(2, 0, 1).copy())
        return tensor, self.class_to_id[row["class_name"]], index


def safe_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", required=True, type=Path, dest="zip_path")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") == args.split]
    if not rows:
        raise SystemExit(f"No {args.split} rows found in manifest.")
    for row in rows:
        row["archive_path"] = row["archive_path"].replace("\\", "/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    class_to_id = {name: i for i, name in enumerate(classes)}
    dataset = TestImages(rows, class_to_id, checkpoint["image_size"], args.zip_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SorghumCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    with zipfile.ZipFile(args.zip_path) as archive, torch.no_grad():
        for images, labels, indices in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            confidences, predictions = probs.max(dim=1)
            for label, pred, confidence, index in zip(labels.tolist(), predictions.tolist(),
                                                       confidences.tolist(), indices.tolist()):
                if label == pred:
                    continue
                row = rows[index]
                original_path = row["archive_path"]
                filename = Path(original_path).name
                output_name = (f"true-{safe_name(classes[label])}__pred-{safe_name(classes[pred])}__"
                               f"conf-{confidence:.3f}__{filename}")
                (args.out_dir / output_name).write_bytes(archive.read(original_path))
                exported.append({"archive_path": original_path, "true_class": classes[label],
                                 "predicted_class": classes[pred], "confidence": f"{confidence:.4f}",
                                 "exported_file": output_name})

    report = args.out_dir / f"misclassified_{args.split}_images.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["archive_path", "true_class", "predicted_class",
                                               "confidence", "exported_file"])
        writer.writeheader()
        writer.writerows(exported)
    print(f"{args.split.title()} images reviewed: {len(rows)}")
    print(f"Misclassified images exported: {len(exported)}")
    print(f"Review images and CSV: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
