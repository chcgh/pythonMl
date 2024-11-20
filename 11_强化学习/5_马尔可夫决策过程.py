import numpy as np
from common import join
from common import solve_valueFunction

np.random.seed(0)


S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合

# 状态转移函数
P = {
    "s1-保持 s1-s1": 1.0, "s1-前往s2-s2": 1.0,
    "s2-保持 s1-s1": 1.0, "s2-前往s3-s3": 1.0,
    "s3-保持 s4-s4": 1.0, "s3-前往s5-s5": 1.0,
    "s4-保持 s5-s5": 1.0, "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4, "s4-概率前往-s4": 0.4,
}

# 奖励函数
R = {
    "s1-保持 s1": -1, "s1-前往 s2": 0,
    "s2-前往 s1": -1, "s2-前往 s3": -2,
    "s3-保持 s4": -2, "s3-前往 s5": 0,
    "s4-保持 s5": 10, "s4-概率前往": 1,
}

gamma = 0.5
MDP = (S, A, P, R, gamma)

# 策略1：随机策略
Pi_1 = {
    "s1-保持 s1": 0.5, "s1-前往 s2": 0.5,
    "s2-前往 s1": 0.5, "s2-前往 s3": 0.5,
    "s3-前往 s4": 0.5, "s3-前往 s5": 0.5,
    "s4-前往 s5": 0.5, "s4-概率前往": 0.5,
}

# 策略1
Pi_2 = {
    "s1-保持 s1": 0.6, "s1-前往 s2": 0.4,
    "s2-前往 s1": 0.3, "s2-前往 s3": 0.7,
    "s3-前往 s4": 0.5, "s3-前往 s5": 0.5,
    "s4-前往 s5": 0.1, "s4-概率前往": 0.9,
}


P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]

P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

V = solve_valueFunction(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma)
print("MDP中每个状态价值分别为\n", V)
