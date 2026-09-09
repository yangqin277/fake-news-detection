## Baseline 1


Model:

TF-IDF + Logistic Regression


Dataset:

Weibo21


Split:

train_test_split 80/20


Feature:

TF-IDF
max_features=5000


Classifier:

Logistic Regression


Result:

Accuracy:

0.7576


Macro F1:

0.7855





---


## Baseline 2

### Model

TF-IDF + Logistic Regression


### Feature

category + content


### Result

Accuracy:

0.7946


F1:

0.7897



### Observation

Adding domain information improves performance.

This indicates that domain-aware information is useful for fake news detection.


## Baseline 3

### Model

Domain-specific TF-IDF + Logistic Regression


### Motivation

Different news domains may contain different linguistic patterns.

Therefore, independent classifiers are trained for each domain.


### Results


| Domain | Accuracy | F1 |
|---|---|---|
| Military |0.878|0.912|
| Politics|0.825|0.877|
| Medical|0.775|0.821|
...


### Observation

Performance varies significantly across domains.

Some domains achieve strong detection performance, while entertainment and finance domains remain challenging.

This indicates domain shift is an important problem for fake news detection.

Low-performing domains may contain more diverse writing styles and implicit misinformation patterns, making them harder for lexical-based models to distinguish.


## MacBERT

Model:
Chinese MacBERT-base

Feature:
Fine-tuned contextual representation


Result:

| Model | Accuracy | F1 |
|-|-|-|
| TF-IDF + LR |0.795|0.790|
| MacBERT |0.902|0.904|


Observation:

MacBERT significantly improves over TF-IDF baseline,
showing that contextual semantic representation is effective
for fake news detection.

## Evaluation

Test set:

1151 samples


Result:

| Metric | Score |
|-|-|
| Accuracy | 0.8879 |
| F1-score | 0.8828 |


Classification Report:

|Label|Precision|Recall|F1|
|-|-|-|-|
|0|0.88|0.91|0.89|
|1|0.90|0.87|0.88|


Confusion Matrix:

[[536,56],
 [73,486]]

 

### Observation

MacBERT achieves strong performance on fake news detection.

The model can effectively capture semantic information from news texts.
However, some false positives and false negatives remain, indicating that
the model may still be affected by writing style and domain distribution.

## Conclusion

Experimental results show that MacBERT achieves better performance than
traditional TF-IDF based methods.

The improvement indicates that contextual semantic representation is useful
for fake news detection.

However, domain differences and implicit misinformation patterns remain
challenging problems.