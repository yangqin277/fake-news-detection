import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# =====================
# 配置
# =====================

DATA_PATH = "data/processed/train.csv"

MODEL_NAME = "hfl/chinese-macbert-base"

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =====================
# Dataset
# =====================

class NewsDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer


    def __len__(self):

        return len(self.texts)


    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]


        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )


        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label)
        }


        return item



# =====================
# Train
# =====================

def main():

    print("Device:", DEVICE)


    # 读取数据

    df = pd.read_csv(DATA_PATH)


    texts = df["content"].tolist()
    labels = df["label"].tolist()


    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )


    print(
        "Train:",
        len(train_texts),
        "Val:",
        len(val_texts)
    )


    # tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    train_dataset = NewsDataset(
        train_texts,
        train_labels,
        tokenizer
    )


    val_dataset = NewsDataset(
        val_texts,
        val_labels,
        tokenizer
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )


    # model

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )


    model.to(DEVICE)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )


    # =====================
    # training loop
    # =====================

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0


        for batch in train_loader:


            batch = {
                k:v.to(DEVICE)
                for k,v in batch.items()
            }


            outputs = model(**batch)


            loss = outputs.loss


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            total_loss += loss.item()



        print(
            f"Epoch {epoch+1}/{EPOCHS}",
            "Loss:",
            total_loss / len(train_loader)
        )


        # validation

        model.eval()


        preds = []
        targets = []


        with torch.no_grad():

            for batch in val_loader:


                labels = batch["labels"]


                batch = {
                    k:v.to(DEVICE)
                    for k,v in batch.items()
                }


                outputs = model(**batch)


                prediction = torch.argmax(
                    outputs.logits,
                    dim=1
                )


                preds.extend(
                    prediction.cpu().numpy()
                )

                targets.extend(
                    labels.numpy()
                )


        acc = accuracy_score(
            targets,
            preds
        )

        f1 = f1_score(
            targets,
            preds
        )


        print(
            "Validation:",
            "Accuracy=",
            acc,
            "F1=",
            f1
        )


    print("Training finished")

    SAVE_PATH = "results/macbert"


    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)


    print("model saved")



if __name__ == "__main__":
    main()

