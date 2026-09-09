import pickle
from pathlib import Path


RAW_PATH = Path("data/raw/train.pkl")
SAVE_PATH = Path("data/processed/train.csv")


def main():

    with open(RAW_PATH,"rb") as f:
        df = pickle.load(f)


    print(df.head())


    SAVE_PATH.parent.mkdir(
        exist_ok=True
    )

    df.to_csv(
        SAVE_PATH,
        index=False,
        encoding="utf-8-sig"
    )


    print("saved:", SAVE_PATH)


if __name__ == "__main__":
    main()