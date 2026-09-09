import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, classification_report


DATA_PATH = "data/processed/train.csv"


def main():

    # 读取数据
    df = pd.read_csv(DATA_PATH)


    X = df["category"] + " " + df["content"]
    y = df["label"]


    # 划分训练/验证
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000
    )


    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)


    # 分类器
    model = LogisticRegression(
        max_iter=1000
    )


    model.fit(
        X_train_tfidf,
        y_train
    )


    # 预测
    y_pred = model.predict(
        X_test_tfidf
    )


    print("Accuracy:")
    print(
        accuracy_score(
            y_test,
            y_pred
        )
    )


    print("\nF1:")
    print(
        f1_score(
            y_test,
            y_pred
        )
    )


    print("\nReport:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )


if __name__ == "__main__":
    main()