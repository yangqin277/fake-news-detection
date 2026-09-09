import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = "data/processed/train.csv"

MODEL_PATH = "results/macbert"


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def main():

    # =====================
    # 读取数据
    # =====================

    df = pd.read_csv(DATA_PATH)

    texts = df["content"].tolist()
    labels = df["label"].tolist()


    # 和训练保持一致
    _, test_texts, _, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )


    print("Test samples:", len(test_texts))


    # =====================
    # 加载模型
    # =====================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(DEVICE)

    model.eval()


    # =====================
    # 推理
    # =====================

    predictions = []


    with torch.no_grad():

        for text in test_texts:

            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )


            inputs = {
                k:v.to(DEVICE)
                for k,v in inputs.items()
            }


            outputs = model(**inputs)


            pred = torch.argmax(
                outputs.logits,
                dim=1
            )


            predictions.append(
                pred.item()
            )


    # =====================
    # 指标
    # =====================

    acc = accuracy_score(
        test_labels,
        predictions
    )

    f1 = f1_score(
        test_labels,
        predictions
    )


    print("\nAccuracy:", acc)
    print("F1:", f1)


    print("\nClassification Report:")
    print(
        classification_report(
            test_labels,
            predictions
        )
    )


    # =====================
    # 混淆矩阵
    # =====================

    cm = confusion_matrix(
        test_labels,
        predictions
    )


    print("Confusion Matrix:")
    print(cm)


    plt.figure(figsize=(5,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Prediction")
    plt.ylabel("True")

    plt.title(
        "MacBERT Confusion Matrix"
    )

    plt.savefig(
        "results/macbert_confusion_matrix.png"
    )


if __name__ == "__main__":
    main()