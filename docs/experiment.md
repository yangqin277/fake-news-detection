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