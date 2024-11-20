import numpy as np

from common import join
from common import MDP


def sample(MDP, Pi, timestep_max, sample_number):
    '''采样函数，策略Pi，限制最长时间步timestep_max，总共采样序列数sample_number'''
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(sample_number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  # 随机选择一个除s5以外的状态s作为起点
        # 当前状态为终止状态或者时间步太长时，一次采样结束
        while s != "s5" and timestep < timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            # 在状态s下根据策略选择动作
            flag = False
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    flag = True
                    break
            if not flag: continue
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态s_next
            flag = False
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    flag = True
                    break
            if not flag: continue
            episode.append((s, a, r, s_next))  # 把(s,a,r,s_next)元组放入序列中
            s = s_next  # s_next变成当前状态，开始接下来的循环
        episodes.append(episode)
    return episodes


mdp = MDP()
# 采样5次，每个序列最长不超过1000步
episodes = sample(mdp.MDP, mdp.Pi_1, 1000, 5)
print('第一条序列\n', episodes[0])
print('第二条序列\n', episodes[1])
print('第五条序列\n', episodes[4])


# 对所有采样序列计算所有状态的价值
def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1):  # 一个序列从后往前计算
            (s, a, r, s_next) = episode[i]
            G = r + gamma + G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]


timestep_max = 20
# 采样1000次
sample_number = 1000
episodes = sample(mdp.MDP, mdp.Pi_1, timestep_max, sample_number)
gamma = 0.5
V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
MC(episodes, V, N, gamma)
print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)
