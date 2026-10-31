# Sorghum Disease Detection Using Deep Learning

An image-based research prototype that predicts one of six Sorghum disease labels from a leaf or head image. The project uses a custom convolutional neural network (CNN) trained from random initialization; it does not use pretrained model weights. Image classification is the method the model uses to detect a disease category.

## Project status

The current candidate was trained on 7,167 readable JPEG images using a stratified image-level split. Its best recorded validation result was 85.5% accuracy and 0.822 macro-F1. These are research results, not a confirmed field diagnosis. The test set was reviewed during development, and the data does not identify plants or photo sessions, so an independent field evaluation is still needed.

The archive has six folder labels: Anthracnose and Red Rot, Cereal Grain molds, Covered Kernel smut, Head Smut, loose smut, and Rust. The combined Anthracnose and Red Rot folder remains a single label in this model.

## Repository contents

- `audit_sorghum.py` — checks the ZIP archive and creates a reproducible train/validation/test manifest.
- `train_sorghum.py` — trains the custom CNN from scratch.
- `predict_sorghum.py` — predicts the top disease labels for one image.
- `inspect_sorghum_errors.py` — exports misclassified images for visual review.
- `find_near_duplicates.py` — finds potential near-duplicate image pairs for review.
- `app.py` — local Streamlit image upload interface.

Dataset archives, image files, generated reports, review exports, and model checkpoints are excluded from Git. The trained checkpoint is available in the [v1.0.0 GitHub Release](https://github.com/Prasad-Pitke/Sorghum-Disease-Detection-Using-Deep-Learning/releases/download/v1.0.0/sorghum_cnn_best.pt). Download the dataset separately from [Mendeley Data](https://data.mendeley.com/datasets/fgb3cmfrg2/1).

## Setup

Use Python 3.10 or newer. From PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the CPU or CUDA build of PyTorch appropriate for your computer if the default `torch` install does not match your hardware.

## Audit and split the dataset

The audit reads the ZIP directly and does not extract or alter it. Replace the example ZIP path with its actual location:

```powershell
python audit_sorghum.py "C:\path\to\Sorghum Disease Image Dataset.zip" --out-dir "sorghum_audit"
```

The generated `sorghum_audit\manifest.csv` records the split for every image. Keep this manifest unchanged when comparing training runs.

## Train the model

```powershell
python train_sorghum.py `
  --zip "C:\path\to\Sorghum Disease Image Dataset.zip" `
  --manifest "sorghum_audit\manifest.csv" `
  --out-dir "results_224_rust125" `
  --epochs 11 `
  --image-size 224 `
  --rust-weight-multiplier 1.25 `
  --seed 42
```

The checkpoint is written under `results_224_rust125`. Training on CPU can take a long time.

## Download and use the trained model

Download [sorghum_cnn_best.pt from the v1.0.0 release](https://github.com/Prasad-Pitke/Sorghum-Disease-Detection-Using-Deep-Learning/releases/download/v1.0.0/sorghum_cnn_best.pt) and save it as `results_224_rust125\sorghum_cnn_best.pt` in the project folder. You can then use the web interface without training the model again.

## Run the web interface

After downloading the checkpoint above or training it into the same output folder:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit and upload a JPEG or PNG image. The displayed scores are not calibrated probabilities or a confirmed diagnosis.

## Citation and dataset

Dataset: [Sorghum Disease Image Dataset, Mendeley Data](https://data.mendeley.com/datasets/fgb3cmfrg2/1). Please check the dataset record for its current citation and terms of use.
