import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "results/macbert"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 这里先沿用你当前项目里的定义：
# 0 = 真实新闻
# 1 = 虚假新闻
# 如果之后确认 Weibo21 的标签定义相反，只需要交换下面两项。
LABEL_MAP = {
    0: "真实新闻",
    1: "虚假新闻"
}


def load_model():
    print("正在加载模型...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(DEVICE)
    model.eval()

    print(f"模型加载完成，当前设备：{DEVICE}")

    return tokenizer, model


def predict(text, tokenizer, model):

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

        predicted_label = torch.argmax(
            probabilities
        ).item()

    return (
        predicted_label,
        probabilities.cpu().tolist()
    )


def main():

    tokenizer, model = load_model()

    print("\n==============================")
    print(" 中文虚假新闻检测 Demo")
    print("==============================")
    print("输入新闻文本后按 Enter 进行检测")
    print("输入 q 退出程序")

    while True:

        print()
        text = input("请输入新闻：").strip()

        if text.lower() == "q":
            print("程序已退出。")
            break

        if not text:
            print("输入不能为空。")
            continue

        label, probabilities = predict(
            text,
            tokenizer,
            model
        )

        confidence = probabilities[label]

        print("\n--------- 检测结果 ---------")
        print("预测结果：", LABEL_MAP[label])
        print(f"置信度：{confidence:.2%}")
        print(f"真实新闻概率：{probabilities[0]:.2%}")
        print(f"虚假新闻概率：{probabilities[1]:.2%}")
        print("----------------------------")


if __name__ == "__main__":
    main()