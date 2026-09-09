import pandas as pd


df=pd.read_csv(
    "data/processed/train.csv"
)


for domain in df["category"].unique():

    sub=df[df["category"]==domain]

    print("\n======")
    print(domain)

    print("samples:",
          len(sub))

    print(
        sub["label"].value_counts()
    )