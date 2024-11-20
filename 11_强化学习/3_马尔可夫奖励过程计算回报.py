# 已知进入每个状态的奖励，计算一个状态序列的回报
# 每个状态的奖励为rewards
# 状态序列为chain
import numpy as np

rewards = [-1, -2, -2, 10, 1, 0]  # 定义到达6个状态分别对应的奖励(奖励函数)
gamma = 0.5


def compute_return(chain):
    G = 0
    for i in range(len(chain)):
        G = G + np.power(gamma, i) * rewards[chain[i] - 1]
    return G


if __name__ == '__main__':
    # chain = [1, 2, 3, 6]
    chain = [1, 1, 2, 3, 4, 5, 3, 6]
    G = compute_return(chain)
    print("回报为:%s。" % G)
