# Synthetic Source Attribution Challenge

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge/overview)

We present a robust framework for the task of synthetic image source attribution, aiming to identify the specific generative model used to create a synthetic image. We propose the Spectral-HELIX architecture, which combines a state-of-the-art spatial backbone with a learnable spectral filtering branch. By analyzing both spatial features and frequency-domain artifacts, the model can capture the "fingerprints" left by different generative sources. We investigate the impact of various backbones (ResNet50, ConvNeXt, EfficientNetV2), the role of image resolution, and the effect of advanced regularization techniques such as Mixup and Exponential Moving Average (EMA). Our final ensemble approach achieves a high generalization accuracy, demonstrating that diversifying model architectures is key to overcoming the performance ceiling.

## Proposed Methodology
### 1 The Spectral-HELIX Architecture
- **Spatial Path:** Utilizes a deep CNN (ConvNeXt/EfficientNet) to capture structural and textural patterns.
- **Spectral Path:** Implements a GlobalFilterBlock based on the Fast Fourier Transform (FFT). This branch transforms the features into the frequency domain, applies a learnable complex-valued filter to isolate generative artifacts, and transforms them back to the spatial domain.
### 2 Dynamic Gated Fusion
Instead of a simple average, we implemented a Dynamic Gating Network. This small MLP analyzes the feature map and predicts a weight $α$ for each image, deciding whether the spatial or spectral path is more reliable for that specific sample.
### 3 Training Recipe (Regularization)
- **EMA (Exponential Moving Average):** Maintained a shadow copy of model weights to smooth out training noise.
- **Mixup:** Synthetically expanded the dataset by interpolating pairs of images and labels, preventing the model from "memorizing" specific training samples.
- **Stratified K-Fold:** Used 5-fold cross-validation to ensure the stability of the results and avoid data leakage.
- **TTA (Test-Time Augmentation):** Averaged predictions across multiple flips and rotations during inference.
  
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
