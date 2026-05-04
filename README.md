# Synthetic Source Attribution Challenge

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge/overview)


This repository contains a professional implementation of an image classification pipeline designed for the **Synthetic Source Attribution Challenge**. The goal is to identify the origin of synthetic images with high precision using deep learning.

## Project Overview

Spectral-HELIX is a high-performance deep learning pipeline designed for Synthetic Source Attribution—the task of identifying the specific generative model (e.g., GAN or Diffusion) that produced a synthetic image.

The framework leverages a hybrid architecture that combines a ConvNeXt-Tiny backbone for high-level spatial feature extraction and a custom Spectral Branch to detect subtle frequency-domain artifacts (fingerprints) left by generative processes. A Dynamic Gating Network intelligently fuses these two pathways, allowing the model to adaptively weight spatial vs. spectral evidence for every individual image.

## Technical Methodology

### 1. Hybrid Spectral-Spatial Architecture
The model employs two parallel pathways:
- **Spatial Path:** A **ConvNeXt-Tiny** backbone extracting high-level semantic and textural features.
- **Spectral Path:** A custom branch utilizing **Learnable Fourier Filters**. It transforms features into the frequency domain to identify periodic artifacts.

### 2. Dynamic Gating Mechanism
To optimize the fusion of the two paths, a **Dynamic Gating Network** was implemented. Instead of static averaging, the model adaptively calculates the trust weight for each path based on the image features:

$$ \text{Final Logits} = g \cdot \text{SpectralLogits} + (1 - g) \cdot \text{SpatialLogits} $$
$$\text{where } g = \sigma(\text{GatingNet}(\text{features}))$$

### 3. Training and Regularization
To achieve a validation accuracy of **97.7%**, the following techniques were utilized:
- **EMA (Exponential Moving Average):** $\theta_{EMA}^{(t)} = \beta \theta_{EMA}^{(t-1)} + (1 - \beta) \theta^{(t)}$, where $\beta=0.999$.
- **Stratified K-Fold Cross-Validation:** Ensuring robustness across $K=6$ folds.
- **Label Smoothing:** Modifying targets to prevent over-confidence: $y_{ls} = y(1 - \alpha) + \frac{\alpha}{K}$.
- **Advanced TTA (Test-Time Augmentation):** Averaging predictions across 5 augmented versions of each test image.

## Data Acquisition

Since the dataset is too large to be hosted on GitHub, you must download it from Kaggle (using Kaggle API).

```bash
pip install kaggle
kaggle competitions download -c synthetic-source-attribution-challenge
```

## Project Structure
The project is organized into a modular structure to ensure maintainability and scalability:
```bash
synthetic-source-attribution/
├── data/               # (Excluded from Git) Put your CSVs and images here
│   └── .gitkeep        # Keeps the folder structure in the repo
├── src/                # Source code
│   ├── __init__.py
│   ├── config.py       # Centralized hyperparameters and file paths
│   ├── dataset.py      # ImageDataset class and transform pipelines
│   ├── model.py        # Model initialization and loading
│   ├── trainer.py      # Training loops and evaluation logic
│   └── inference.py    # Ensemble prediction and TTA logic
├── main.py             # Main entry point to run the entire pipeline
├── .gitignore          # Prevents uploading large datasets/weights
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## How to Run
- Clone the Repository
```bash
git clone https://github.com/Alihcyv/Synthetic-Image-Attribution.git
cd Synthetic-Image-Attribution
```
- 2. Install Dependencies
```bash
pip install -r requirements.txt
```
- Data Setup:
Please organize your `data/` folder as follows:
```bash
data/
├── training.csv
├── test.csv
└── images/
    ├── img_1.jpg
    └── ...
```
Note: Update the paths in src/config.py to match your local environment if you use a different directory.
- Execute the Pipeline
```bash
python main.py
```
