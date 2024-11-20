import random
import pandas as pd


def demo1():
    # 第二个元素越大，采到的概率越大，请采样1w次，然后统计频率
    a = [["A", 10], ["B", 20], ["C", 30], ["D", 40]]
    ra = random.choices(population=[i[0] for i in a], weights=[i[1] for i in a], k=100000)
    print(pd.Series(ra).value_counts())


def demo2():
    ra = random.choices(population=["A", 'B', 'C', 'D'], weights=[10, 20, 30, 40], k=10000)
    print(pd.Series(ra).value_counts())



if __name__ == '__main__':
    # demo1()
    demo2()
