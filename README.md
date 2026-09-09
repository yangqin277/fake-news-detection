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

### Validation Performance

| Model | Accuracy | F1 |
|-|-|-|
| TF-IDF + LR |0.795|0.790|
| MacBERT |0.902|0.904|


### Test Performance

| Model | Accuracy | F1 |
|-|-|-|
| MacBERT |0.8879|0.8828|


## Environment

Python

PyTorch

Transformers


## Usage

Train:

```bash
python src/train_macbert.py
```
Predict:

```bash
python src/test_macbert.py
```

Demo:

```bash
python src/demo.py
```

## Evaluation

Confusion Matrix:

![MacBERT Confusion Matrix](results/macbert_confusion_matrix.png)

## Demo

Example:

Input:

专家称喝可乐可以治疗癌症


Output:

Prediction:

虚假新闻

Confidence:

98.60%

