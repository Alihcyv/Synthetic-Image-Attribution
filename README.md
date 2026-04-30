# 🖼️ Synthetic Source Attribution Challenge

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/)

This repository contains a professional implementation of an image classification pipeline designed for the **Synthetic Source Attribution Challenge**. The goal is to identify the origin of synthetic images with high precision using deep learning.

## 🚀 Project Overview
The pipeline implements a state-of-the-art image classification approach utilizing the **ConvNeXt** architecture. To ensure the model's robustness and generalization, I employed a combination of **Stratified K-Fold Cross-Validation**, **Label Smoothing**, and **Test-Time Augmentation (TTA)**.

## 🛠️ Technical Methodology

### 1. Model Architecture
I utilized the $\text{ConvNeXt-Tiny}$ model (pretrained on $\text{ImageNet-22K}$) via the `timm` library. ConvNeXt provides a modern take on convolutional neural networks, offering performance competitive with Vision Transformers (ViTs) while maintaining the efficiency of CNNs.

### 2. Training Strategy
- **Cross-Validation:** A 10-Fold Stratified K-Fold approach was used. The validation accuracy is calculated as the mean of all folds:
$$\text{CV Accuracy} = \frac{1}{K} \sum_{i=1}^{K} \text{Acc}_i$$
- **Regularization:** 
    - **Label Smoothing:** Used to prevent the model from becoming over-confident. The target distribution is modified as:
      $$y_{ls} = y(1 - \alpha) + \frac{\alpha}{K}$$
      where $\alpha=0.1$ and $K$ is the number of classes.
    - **Learning Rate Scheduler:** Implemented $\text{CosineAnnealingWarmRestarts}$ to optimize convergence.
- **Augmentation:** A custom pipeline including $\text{GaussianBlur}$ and $\text{RandomAutocontrast}$ was applied to improve generalization.

### 3. Inference & Ensemble
To maximize the final score, the following techniques were used:
- **Ensembling:** Averaging softmax probabilities from all $K=10$ fold-models.
- **TTA (Test Time Augmentation):** Each test image $x$ was passed through the model in two versions (Original $x$ and Horizontally Flipped $x_{flip}$):
$$\text{Final Prob} = \frac{P(y|x) + P(y|x_{flip})}{2}$$

## 📥 Data Acquisition

Since the dataset is too large to be hosted on GitHub, you must download it from Kaggle (using Kaggle API).

```bash
pip install kaggle
kaggle competitions download -c synthetic-source-attribution-challenge
```

## 📁 Project Structure
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
