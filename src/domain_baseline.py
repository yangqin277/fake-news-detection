import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score


DATA_PATH = "data/processed/train.csv"


def train_domain_model(df, domain):

    print("\nDomain:", domain)


    domain_df = df[df["category"] == domain]


    print(domain_df["label"].value_counts())


    X = domain_df["content"]
    y = domain_df["label"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    vectorizer = TfidfVectorizer(
        max_features=5000
    )


    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)


    model = LogisticRegression(
        max_iter=1000
    )


    model.fit(
        X_train_tfidf,
        y_train
    )


    pred = model.predict(
        X_test_tfidf
    )


    acc = accuracy_score(
        y_test,
        pred
    )

    f1 = f1_score(
        y_test,
        pred
    )


    print("Accuracy:", round(acc,4))
    print("F1:", round(f1,4))


    return acc,f1



def main():

    df = pd.read_csv(DATA_PATH)


    results=[]


    for domain in df["category"].unique():

        acc,f1=train_domain_model(
            df,
            domain
        )

        results.append(
            {
                "domain":domain,
                "accuracy":acc,
                "f1":f1
            }
        )


    result_df=pd.DataFrame(results)

    print("\nSummary")
    print(result_df)


if __name__=="__main__":
    main()