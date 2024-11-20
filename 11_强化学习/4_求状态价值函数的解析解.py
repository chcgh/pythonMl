# 贝尔曼方程：一个状态的价值，等于即时奖励，加上所有下一个状态的价值的加权和

import numpy as np
from common import solve_valueFunction

# 状态转移矩阵
P = np.array([
    [0.9, 0.1, 0, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0, 0],
    [0, 0, 0, 0.6, 0, 0.4],
    [0, 0, 0, 0, 0.3, 0.7],
    [0, 0.2, 0.3, 0.5, 0, 0],
    [0, 0, 0, 0, 0, 1]
])

# 进入各个状态所得到的奖励，即奖励函数
rewards = [-1, -2, -2, 10, 1, 0]
gamma = 0.5

if __name__ == '__main__':
    V = solve_valueFunction(P, rewards, gamma)
    print("MRP中每个状态价值分别为\n", V)
