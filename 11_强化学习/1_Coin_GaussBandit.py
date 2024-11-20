import random
import numpy as np


class Coin:
    def __init__(self):
        self.p = random.random()

    def toss(self):
        tp = random.random()
        return 1 if tp < self.p else 0


def test_Coin():
    coin = Coin()
    count = 0
    N = 100000
    for _ in range(N):
        count += coin.toss()

    print(coin.p)
    print(count / N)


# 高斯老虎机：每个臂获得的奖励符合高斯分布
class GaussBandit:
    def __init__(self):
        self.mean = np.random.uniform(200, 1000)
        self.std = np.random.uniform(10, 60)

    def play(self):
        return np.random.normal(self.mean, self.std)


def test_Bandit():
    bandit = GaussBandit()
    result = []
    for _ in range(100000):
        result.append(bandit.play())

    print(np.mean(result), np.std(result))
    print(bandit.mean, bandit.std)


if __name__ == '__main__':
    # test_Coin()
    test_Bandit()
