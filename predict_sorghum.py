"""Predict Sorghum disease class for one new image using the saved scratch CNN.

Run:
  python predict_sorghum.py --model "E:/ASA model/results/sorghum_cnn_best.pt" --image "E:/ASA model/new_photo.jpg"

The scores are model confidence scores, not validated disease probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_sorghum import SorghumCNN


def preprocess(image_path: Path, size: int) -> torch.Tensor:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model file not found: {args.model}")
    if not args.image.is_file():
        parser.error(f"Image file not found: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model = SorghumCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tensor = preprocess(args.image, checkpoint["image_size"]).to(device)
    with torch.no_grad():
        scores = torch.softmax(model(tensor), dim=1)[0]
    values, indices = torch.topk(scores, k=min(3, len(classes)))
    print(f"Image: {args.image}")
    print("Top predictions:")
    for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        print(f"  {rank}. {classes[index]} — {score:.1%}")
    print("These are model confidence scores, not calibrated probabilities or a confirmed diagnosis.")


if __name__ == "__main__":
    main()
