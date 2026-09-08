import pandas as pd
from pathlib import Path


print("Fake News Detection Data Explorer")


# 数据路径
data_path = Path("data/raw/train.csv")

print("数据路径:")
print(data_path.absolute())


# 判断文件是否存在
if data_path.exists():

    df = pd.read_csv(data_path)

    print("\n数据预览:")
    print(df.head())

    print("\n数据规模:")
    print(df.shape)

    print("\n字段信息:")
    print(df.info())

else:

    print("数据文件不存在:")
    print(data_path)