import pandas as pd
import torch

from torch.utils.data import Dataset

from transformers import AutoTokenizer

from transformers import AutoModelForSequenceClassification


DATA_PATH = "data/processed/train.csv"

MODEL_NAME = "hfl/chinese-macbert-base"


class NewsDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=128):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):

        return len(self.texts)


    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]


        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )


        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "label": torch.tensor(label)
        }



def main():

    df = pd.read_csv(DATA_PATH)


    texts = df["content"].tolist()
    labels = df["label"].tolist()


    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    dataset = NewsDataset(
        texts,
        labels,
        tokenizer
    )


    sample = dataset[0]

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )


    print(model)


    print(sample.keys())

    print(
        sample["input_ids"].shape
    )

    print(
        sample["label"]
    )


if __name__ == "__main__":
    main()