# Aircraft Classification in Satellite Imagery 🛰️

## Project Overview
This project focuses on the automatic classification of military aircraft types from aerial and satellite images using Deep Learning (DL) techniques. The system is designed to support applications in security, airspace monitoring, and technical reconnaissance.

I compare the performance of six diverse neural network architectures (ranging from classic CNNs to modern Transformers) and demonstrate how Ensemble Methods (Soft-Voting and Stacking) can be utilized to significantly improve prediction stability and calibration.

### Key Results
The Stacking Ensemble method achieved the highest overall performance, demonstrating superior accuracy and the best probabilistic calibration (lowest Log-loss) on the test dataset.

| Metoda / Model | Accuracy | F1-score | Log-loss |
| :--- | :--- | :--- | :--- |
| **ConvNeXt-Tiny (Best Single)** | 0.968 | 0.959 | 0.130 |
| **Soft-voting** | 0.969 | 0.961 | 0.113 |
| **Stacking (Logistic Regression)** | **0.970** | **0.963** | **0.105** |

*Best single model: **ConvNeXt-Tiny**.*
*Best overall result: **Stacking**.*

---

## Setup and Installation

### Prerequisites
The project is implemented in **PyTorch** and requires Python 3.11. I recommend using a virtual environment (e.g., Anaconda or Miniconda).

```bash
# Create and activate environment
conda create -n aircraft_clf python=3.11
conda activate aircraft_clf

# Install necessary libraries
# Ensure PyTorch supports your CUDA version if using a GPU
pip install torch torchvision torchaudio 
pip install scikit-learn wandb numpy pandas tqdm
```
---

## Data

The project utilizes the public [**MAR20 (Military Aircraft Recognition)**](https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset) dataset. Aircraft objects were cropped according to annotations and scaled to a fixed resolution of $224 \times 244 pixels$.

---

## Repository Structure and Scripts

The project workflow is managed by five primary scripts located in the `scripts/` directory.

| Script | Description | Purpose (Stage) |
| :--- | :--- | :--- |
| `01_split_dataset.py` | **Data Preparation and Split.** Deterministically splits input class folders into `train`, `val`, and `test` directories. | Preprocessing |
| `02_train.py` | **Model Training.** Implements transfer learning, training one model (e.g., ConvNeXt-Tiny) at a time, using W&B logging and early stopping. | Training |
| `03_ensemble_classifiers.py` | **Ensemble Building.** Creates Soft-Voting and Stacking ensembles from prediction logs of individual models (requires `sklearn` for stacking). | Ensemble |
| `04_evaluate.py` | **Model Evaluation.** Assesses performance of single models and simple ensembles (`prob_avg`, `logits_avg`, `majority`) on the test set, reporting macro metrics and Confusion Matrices. | Evaluation |
| `05_predict_image.py` | **Single-Image Inference.** Predicts the class and Top-K probabilities for a single image using a model or ensemble. | Inference |

---

## Workflow and Usage Instructions

The workflow proceeds sequentially from data preparation to final prediction.

### 1. Data Preparation (`01_split_dataset.py`)

This script creates the `train/`, `val/`, and `test/` splits required for deep learning training.

```bash
# Example: 70/15/15 split using 'copy' mode
python scripts/01_split_dataset.py \
    --input data/MAR20 \
    --output data/processed \
    --train 0.70 --val 0.15 --test 0.15 --mode copy
```
### 2. Model Training (`02_train.py`)

Train each of the six required deep learning architectures. The script handles transfer learning and saves checkpoints (`best.pt`, `last.pt`).

```bash
# Example: Training the ConvNeXt-Tiny model
python scripts/02_train.py \
    --data_dir data/processed \
    --model convnext_tiny \
    --img_size 224 \
    --lr 1e-4 \
    --batch_size 32 \
    --epochs 30 \
    --patience 7 \
    --project aircraft-classification \
    --run_name ConvNeXt-Tiny-Run 
    # NOTE: Repeat this step for VGG16, DenseNet121, MobileNetV3-Large, EfficientNet-B1, and ViT-B/16.
```

### 3. Ensemble Building (`03_ensemble_classifiers.py`)

This script requires the raw probability/logit outputs of the individual models on a validation set (typically obtained during training or a separate prediction step) to train the **Stacking** meta-classifier and configure **Soft-Voting**.

```bash
# Example: Build Stacking and Soft-Voting from pre-generated logs 

python scripts/03_ensemble_classifiers.py \
    --inputs model_logs/convnext_val.csv model_logs/densenet_val.csv ... \
    --id-col filename --class-cols A1 A2 A3 A4 ... \
    --y-col label \
    --strategy stacking \
    --output ensembles/stacking_pred.csv
    
# To use Soft-Voting (weighted by F1-score as in the paper [cite: 94]):
python scripts/03_ensemble_classifiers.py \
    --inputs model_logs/convnext_val.csv model_logs/densenet_val.csv ... \
    --id-col filename --class-cols A1 A2 A3 A4 ... \
    --y-col label \
    --strategy weighted-soft \
    --optimize-weights # Optimizes weights based on validation accuracy/loss
    --output ensembles/soft_voting_pred.csv
```

### 4. Evaluation (`04_evaluate.py`)

Assess the final performance of a single best model or a simple ensemble (prob_avg, logits_avg, majority) on the separate test set.

```bash
# Example 1: Evaluate the best single model (ConvNeXt-Tiny)
python scripts/04_evaluate.py \
    --data_dir data/processed \
    --models_dir ./outputs/ConvNeXt-Tiny-Run \
    --select "best.pt" \
    --ensemble none

# Example 2: Evaluate a simple Logits-Averaging ensemble over the top-4 models
python scripts/04_evaluate.py \
    --data_dir data/processed \
    --models_dir ./outputs \
    --select "*/best.pt" \
    --topk 4 \
    --ensemble logits_avg
```
### 5. Single Image Prediction (`05_predict_image.py`)

Use a trained model or a configured ensemble (e.g., Stacking configuration from 03_ensemble_classifiers.py) to classify a new, single image.

```bash
# Example 1: Predict using the best single model (ConvNeXt-Tiny)
python scripts/05_predict_image.py \
    --image path/to/new_aircraft_image.jpg \
    --models_dir ./outputs/ConvNeXt-Tiny-Run \
    --select "best.pt" \
    --model_fallback convnext_tiny \
    --k 3

# Example 2: Predict using an ensemble (e.g., simple probability average)
python scripts/05_predict_image.py \
    --image path/to/another_aircraft_image.jpg \
    --models_dir ./outputs \
    --select "*/best.pt" \
    --ensemble prob_avg \
    --k 3
```
