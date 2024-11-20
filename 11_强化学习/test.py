
import numpy as np
import random

class Coin():
    def __init__(self):
        self.p = random.random()
        print(self.p)
    def toss(self):
        if random.random()<self.p:
            return 1
        else:
            return 0



if __name__ == '__main__':
    n = 100000
    coin = Coin()
    count = 0
    for _ in range(n):
        r = coin.toss()
        count += r
    print(count/n)
