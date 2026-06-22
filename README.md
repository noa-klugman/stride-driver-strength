# STRIDE: Structure-informed prediction of cancer driver mutation strength

This repository contains the code and data needed to train the STRIDE driver-strength classifier and apply it to AlphaMissense pathogenic variants.

STRIDE classifies cancer missense variants into two categories:

* **Weak driver**
* **Strong driver**

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate stride_training
```

## 1. Train the STRIDE model

Run:

```bash
python code/01_train_stride_model.py
```

This script trains the STRIDE classifier using the TVA-supported training variants in:

```text
raw_data/stride_training_variants_features.csv
```

The script scans TVA thresholds, tunes Random Forest models, selects the best TVA threshold according to validation balanced accuracy, evaluates the final model, and saves the results.

## 2. Predict STRIDE labels for AlphaMissense variants

After training the model, run:

```bash
python code/02_predict_alphamissense.py
```

This script applies the trained STRIDE model to:

```text
raw_data/alphamissense_pathogenic_data.csv
```

and saves the predicted weak/strong labels.

## Notes

The training and prediction datasets already include the required structural and evolutionary features.

The code uses grouped splitting by gene and amino-acid position, so variants from the same protein position are kept in the same split.

The random seed is fixed to:

```python
RANDOM_STATE = 42
```

for reproducibility.
