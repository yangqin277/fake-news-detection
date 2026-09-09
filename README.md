# Fake News Detection Based on MacBERT


## Introduction

This project focuses on Chinese fake news detection.

We compare traditional machine learning methods with
pretrained language models.


## Dataset

Dataset:
Weibo21


## Methods

Baseline:

TF-IDF + Logistic Regression


Advanced Model:

Chinese MacBERT fine-tuning


## Results

| Model | Accuracy | F1 |
|-|-|-|
| TF-IDF + LR |0.795|0.790|
| MacBERT |0.888|0.883|


## Environment

Python

PyTorch

Transformers


## Usage

Train:

```bash
python src/train_macbert.py

```
Predict：

```bash
python src/test_macbert.py

```
## Evaluation

Confusion Matrix:

![MacBERT Confusion Matrix](results/macbert_confusion_matrix.png)

