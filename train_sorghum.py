"""Train a small CNN from scratch using audit_sorghum.py's manifest.

Install: python -m pip install torch pillow numpy
Run: python train_sorghum.py --zip "C:/Users/USER/Downloads/Sorghum Disease Image Dataset.zip" --manifest "E:/ASA model/sorghum_audit/manifest.csv" --out-dir "E:/ASA model/results"
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SorghumDataset(Dataset):
    def __init__(self, rows: list[dict], class_to_id: dict[str, int], size: int,
                 train: bool, archive_path: Path):
        self.rows = rows
        self.class_to_id = class_to_id
        self.size = size
        self.train = train
        self.archive_path = archive_path
        self._zip = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        # Open on demand: no need to extract the 1.38 GB archive.
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.archive_path)
        data = self._zip.read(row["archive_path"])
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            image.thumbnail((self.size, self.size), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
            canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))

        if self.train:
            if random.random() < 0.5:
                canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                canvas = canvas.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            canvas = canvas.rotate(random.uniform(-12, 12), fillcolor=(114, 114, 114))
            if random.random() < 0.8:
                canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.85, 1.15))
                canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.85, 1.15))

        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy())
        return tensor, self.class_to_id[row["class_name"]]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class SorghumCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32), ConvBlock(32, 64), ConvBlock(64, 128), ConvBlock(128, 192),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(192, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("split") in {"train", "validation", "test"}]
    if not rows:
        raise SystemExit("No valid rows found in manifest.")
    return rows


def evaluate(model, loader, device, loss_fn, class_count):
    model.eval()
    loss_total = 0.0
    total = correct = 0
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += loss_fn(logits, labels).item() * labels.size(0)
            preds = logits.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for actual, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                matrix[actual, pred] += 1
    return loss_total / max(total, 1), correct / max(total, 1), matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path, dest="zip_path")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--rust-weight-multiplier", type=float, default=1.0,
                        help="Multiply Rust's class loss weight; try 1.25 to target Rust recall")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rust_weight_multiplier <= 0:
        parser.error("--rust-weight-multiplier must be greater than 0")
    if not args.zip_path.is_file() or not args.manifest.is_file():
        parser.error("ZIP or manifest path does not exist.")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_manifest(args.manifest)
    classes = sorted({r["class_name"] for r in rows}, key=str.casefold)
    class_to_id = {name: i for i, name in enumerate(classes)}
    splits = {name: [r for r in rows if r["split"] == name] for name in ("train", "validation", "test")}
    if any(not splits[s] for s in splits):
        raise SystemExit("Manifest must contain train, validation, and test rows.")
    for split_rows in splits.values():
        for row in split_rows:
            row["archive_path"] = row["archive_path"].replace("\\", "/")

    # Map archive paths used by the manifest to the original archive entry names.
    with zipfile.ZipFile(args.zip_path) as zf:
        archive_names = set(zf.namelist())
    for row in rows:
        if row["archive_path"] not in archive_names:
            raise SystemExit(f"Manifest path missing from ZIP: {row['archive_path']}")
    def loader(split: str):
        ds = SorghumDataset(splits[split], class_to_id, args.image_size,
                            split == "train", args.zip_path)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"),
                          num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader, val_loader, test_loader = loader("train"), loader("validation"), loader("test")
    model = SorghumCNN(len(classes)).to(device)
    train_counts = Counter(r["class_name"] for r in splits["train"])
    weights = torch.tensor([len(splits["train"]) / (len(classes) * train_counts[c]) for c in classes],
                           dtype=torch.float32, device=device)
    if "Rust" in class_to_id:
        weights[class_to_id["Rust"]] *= args.rust_weight_multiplier
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = args.out_dir / "sorghum_cnn_best.pt"
    history = []
    print(f"Device: {device}; classes: {classes}; counts: {dict(train_counts)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = correct = total = 0
        total_batches = len(train_loader)
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            # Refresh every 10 batches and at the end to show activity without
            # flooding the terminal with a line for every batch.
            if batch_index % 10 == 0 or batch_index == total_batches:
                percent = 100 * batch_index / total_batches
                print(f"\rEpoch {epoch:02}/{args.epochs} | batches {batch_index}/{total_batches} "
                      f"({percent:5.1f}%) | running loss {running_loss / total:.4f} "
                      f"acc {correct / total:.3f}", end="", flush=True)
        val_loss, val_acc, _ = evaluate(model, val_loader, device, loss_fn, len(classes))
        scheduler.step(val_loss)
        train_loss, train_acc = running_loss / total, correct / total
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "validation_loss": val_loss, "validation_accuracy": val_acc})
        print(f"\rEpoch {epoch:02}/{args.epochs} complete | train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "image_size": args.image_size, "seed": args.seed,
                        "rust_weight_multiplier": args.rust_weight_multiplier}, best_path)

    with (args.out_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, matrix = evaluate(model, test_loader, device, loss_fn, len(classes))
    print(f"Test loss: {test_loss:.4f}; test accuracy: {test_acc:.4f}")
    with (args.out_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["actual\\predicted", *classes])
        for name, row in zip(classes, matrix.tolist()): writer.writerow([name, *row])
    with (args.out_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["class", "precision", "recall", "f1", "support"])
        for i, name in enumerate(classes):
            tp = int(matrix[i, i]); fp = int(matrix[:, i].sum() - tp); fn = int(matrix[i, :].sum() - tp)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            writer.writerow([name, f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", int(matrix[i, :].sum())])
    with (args.out_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump({"classes": classes, "device": str(device), "test_loss": test_loss,
                   "test_accuracy": test_acc, "seed": args.seed, "image_size": args.image_size,
                   "rust_weight_multiplier": args.rust_weight_multiplier}, f, indent=2)
    print(f"Saved best model and reports in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
