import pandas as pd

print("Fake News Detection Data Explorer")

# 测试 pandas
data = {
    "text": [
        "这是第一条新闻",
        "这是第二条新闻"
    ],
    "label": [
        0,
        1
    ]
}

df = pd.DataFrame(data)

print(df)

print("\n数据基本信息：")
print(df.info())