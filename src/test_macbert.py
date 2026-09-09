import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_PATH = "results/macbert"


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



def predict(text):

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )


    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )


    model.to(DEVICE)

    model.eval()


    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )


    inputs = {
        k:v.to(DEVICE)
        for k,v in inputs.items()
    }


    with torch.no_grad():

        outputs = model(**inputs)


        probs = torch.softmax(
            outputs.logits,
            dim=1
        )


        pred = torch.argmax(
            probs,
            dim=1
        ).item()



    return pred, probs[0][pred].item()



if __name__ == "__main__":


    text = input(
        "请输入新闻："
    )


    label, confidence = predict(text)


    if label == 1:
        result = "虚假新闻"
    else:
        result = "真实新闻"


    print()
    print("预测结果:", result)
    print(
        "置信度:",
        round(confidence,4)
    )