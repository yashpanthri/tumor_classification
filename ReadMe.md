# Brain Tumor Classification using Autoencoders
**University of Michigan - ECE579**

---

## Project Overview

This project implements an unsupervised brain tumor classification pipeline using:
- Convolutional Autoencoder for feature extraction
- K-Means clustering (K=5) for classification
- SVD preprocessing for dimensionality reduction
- Binary tumor detection with localization

---

## Results Summary

| Model | Optimizer | 4-Class Accuracy | Binary Accuracy | F1 Score |
|-------|-----------|------------------|-----------------|----------|
| **Exp2a (Best)** | Adam | 51.51% | 82.15% | 87.85% |
| **H9 (Best SGD)** | SGD | 49.60% | 74.08% | 83.00% |
| **SVD 95%** | SGD | 49.21% | 73.69% | 82.74% |

---

## Prerequisites

```bash
pip install torch torchvision numpy scikit-learn matplotlib pandas pillow scipy opencv-python --break-system-packages
```

---

## Project Structure

```
IS579/
├── Train/                          # Training images (by class)
├── Test/                           # Test images (by class)
├── Validate/                       # Validation images (by class)
├── new_images/                     # New images for detection
├── outputs_baseline/               # SVD preprocessed datasets
│   ├── svd_95/                     # 95% energy retained
│   └── svd_99/                     # 99% energy retained
├── outputs_exp2a_latent256/        # Best model (Adam, latent=256)
├── outputs_H9_lr001_mom09_b16/     # Best SGD model
├── outputs_SVD95_H9params/         # Best SVD model
├── config.py                       # Configuration settings
├── Autoencoder.py                  # Model architecture
├── train_autoencoder.py            # Training script
├── extract_features.py             # Feature extraction
├── k5_evaluation.py                # 4-class evaluation
├── k5_evaluation_binary.py         # Binary evaluation
├── svd_preprocessing.py            # SVD batch preprocessing
├── binary_tumor_detection.py       # Detection (raw images)
└── binary_tumor_detection_SVD.py   # Detection (SVD images)
```

---

## Configuration

All settings are in `config.py`. Update these with users file directories before running. Example provided from original user:

```python
# Data paths
BASE_DIR = r"C:\Users\********\***"
TRAIN_FOLDER = os.path.join(BASE_DIR, "Train")
RESULTS_DIR = r"C:\Users\********\***\outputs_exp2a_latent256"

# Model parameters
LATENT_DIM = 256        # Latent space dimensions
BATCH_SIZE = 8          # Batch size (8 for Adam, 16 for SGD)
LEARNING_RATE = 0.001   # Learning rate
EPOCHS = 50             # Training epochs

# Clustering
N_CLUSTERS = 5          # K-Means clusters
```

---

## Pipeline Execution

### Train Autoencoder (~45 min)

```bash
python train_autoencoder.py
```

**Output:** `outputs_*/models/autoencoder_best.pth`

---

### Extract Features (~2 min)

```bash
python extract_features.py
```

**Output:** `outputs_*/features/test_features.npy`, `validate_features.npy`

---

### K-Means Clustering (~1 min)

```bash
python k_means_clustering_k5.py
```

**Output:** `outputs_*/features/k5/cluster_centers.npy`, `cluster_assignments.npy`

---

### Evaluation (~2 min)

```bash
# 4-class evaluation
python k5_evaluation.py

# Binary evaluation (Tumor vs No Tumor)
python k5_evaluation_binary.py
```

**Output:** `outputs_*/features/k5/evaluation_report.txt`, confusion matrices

---

## Experiment Configurations

### Experiment 2a: Best Model (Adam)

```python
# config.py settings
RESULTS_DIR = r"C:\Users\******\***\outputs_exp2a_latent256"
TRAIN_FOLDER = os.path.join(BASE_DIR, "Train")  # Raw images
LATENT_DIM = 256
BATCH_SIZE = 8
LEARNING_RATE = 0.001
# Optimizer: Adam (in train_autoencoder.py)
```

---

### Hyperparameter Experiments (H1-H10)

Use `train_hyperparams.py` for automated SGD experiments:

```bash
python train_hyperparams.py
```

| Exp | LR | Momentum | Batch | Results |
|-----|-----|----------|-------|---------|
| H1 | 0.001 | 0.0 | 8 | 73.61% binary |
| H2 | 0.001 | 0.9 | 8 | 71.91% binary |
| H3 | 0.01 | 0.0 | 8 | 72.50% binary |
| H4 | 0.01 | 0.9 | 8 | 64.22% binary |
| H5 | 0.01→decay | 0.0 | 8 | 72.46% binary |
| H6 | 0.01→decay | 0.9 | 8 | 63.71% binary |
| H7 | 0.001 | 0.5 | 8 | 74.08% binary |
| H8 | 0.01 | 0.5 | 8 | 71.19% binary |
| **H9** | **0.001** | **0.9** | **16** | **74.08% binary** |
| H10 | 0.001 | 0.9 | 32 | 73.91% binary |

---

### SVD Preprocessing Experiments

#### Generate SVD Preprocessed Images

```bash
# Demo on single image first
python svd_demo.py

# Batch preprocess all images
python svd_preprocessing.py
```

**Output:** `outputs_baseline/svd_95/` and `outputs_baseline/svd_99/`

#### Train on SVD Images

```python
# config.py settings for SVD 95%
TRAIN_FOLDER = os.path.join(BASE_DIR, "outputs_baseline", "svd_95", "Training")
TEST_FOLDER = os.path.join(BASE_DIR, "outputs_baseline", "svd_95", "Testing")
VALIDATE_FOLDER = os.path.join(BASE_DIR, "outputs_baseline", "svd_95", "Validation")
RESULTS_DIR = r"C:\Users\********\***\outputs_SVD95_H9params"

# Use H9 hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
# Optimizer: SGD with momentum=0.9
```

Then run the full pipeline (Stages 1-4).

| SVD Level | Silhouette | 4-Class | Binary | All 4 Classes? |
|-----------|------------|---------|--------|----------------|
| H9 (Raw) | 0.0855 | 49.60% | 74.08% | ✅ |
| **SVD 95%** | **0.0942** | **49.21%** | **73.69%** | **✅** |
| SVD 99% | 0.0986 | 44.50% | 73.69% | ❌ |

---

## Binary Tumor Detection

### Using Raw Images (H9 or Exp2a model)

```python
# config.py - point to H9 or Exp2a model
RESULTS_DIR = r"C:\Users\********\***\outputs_H9_lr001_mom09_b16"
MODEL_PATH = r"C:\Users\********\***\outputs_H9_lr001_mom09_b16\models\autoencoder_best.pth"
```

```bash
# Place images in new_images folder, then run:
python binary_tumor_detection.py
```

### Using SVD Preprocessed Images (SVD95 model)

```python
# config.py - point to SVD95 model
RESULTS_DIR = r"C:\Users\********\***\outputs_SVD95_H9params"
MODEL_PATH = r"C:\Users\********\***\outputs_SVD95_H9params\models\autoencoder_best.pth"
```

```bash
# Place raw images in new_images folder (SVD applied automatically)
python binary_tumor_detection_SVD.py
```

**Output:** 
- `outputs_*/detection_results/` — Visualizations with bounding boxes
- `detection_report.txt` — Summary report

---

## Key Findings

1. **Adam optimizer outperforms SGD** (+8% binary accuracy)
2. **K=5 clusters better than K=4** (+7% 4-class accuracy)
3. **Latent dim 256 is optimal** (better than 128 or 384)
4. **High LR + High Momentum = Bad** (H4, H6 worst performers)
5. **Batch size 16 optimal for SGD** (H9 best SGD model)
6. **SVD 95% preserves all classes** (SVD 99% loses meningioma)
7. **Meningioma hardest to detect** (best: 19.21% with Adam)

---

## Output Folders

| Folder | Description |
|--------|-------------|
| `outputs_H9_lr001_mom09_b16` | Best SGD (74.08% binary) |
| `outputs_SVD95_H9params` | Best SVD (73.69% binary) |
| `outputs_baseline` | SVD preprocessed datasets |
| `outputs_H1_*` through `outputs_H10_*` | Hyperparameter experiments |
| `outputs_exp1a_*`, `outputs_exp1b_*` | Epoch experiments |
| `outputs_exp2b_*` | Latent dimension experiments |

---

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `config.py`:
```python
BATCH_SIZE = 4  # or 8
```

### Module Not Found
Ensure virtual environment is activated:
```bash
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### Low Detection Accuracy on New Images
The model works best on images similar to training data. Different MRI modalities (T1 vs T2) may produce poor results.

---

## References

- Dataset: Brain Tumor MRI Dataset (Kaggle)
- Architecture: Convolutional Autoencoder with 5 encoder/decoder layers
- Clustering: K-Means with K=5, StandardScaler normalization

---

**Author:** Matthew Sheaffer  
**Course:** ECE579 - University of Michigan  
**Date:** November 2025