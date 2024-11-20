import numpy as np


# 把输入的两个字符串通过"-"连接，便于使用上述定义的P,R变量
def join(str1, str2):
    return str1 + '-' + str2

# 求状态价值函数的解析解
def solve_valueFunction(P, rewards, gamma):
    states_num = P.shape[0]
    rewards = np.array(rewards).reshape((-1, 1))
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value


class MDP:
    def __init__(self):
        self.S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
        self.A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往s2", "概率前往s3",
                  "概率前往s4"]  # 动作集合

        # 状态转移函数
        # "s1-保持s1-s1": 1.0的含义是在状态s1下动作保持在状态s1之后转移到状态s1的概率是1
        self.P = {
            "s1-保持s1-s1": 1.0, "s1-前往s2-s2": 1.0,
            "s2-前往s1-s1": 1.0, "s2-前往s3-s3": 1.0,
            "s3-前往s4-s1": 1.0, "s3-前往s5-s5": 1.0,
            "s4-前往s5-s5": 1.0, "s4-概率前往s2-s2": 0.2,
            "s4-概率前往s3-s3": 0.4, "s4-概率前往s4-s4": 0.4,
        }

        # 奖励函数
        self.R = {
            "s1-保持s1": -1, "s1-前往s2": 0,
            "s2-前往s1": -1, "s2-前往s3": -2,
            "s3-前往s4": -2, "s3-前往s5": 0,
            "s4-前往s5": 10, "s4-概率前往s2": 1,
            "s4-概率前往s3": 1, "s4-概率前往s4": 1,
        }

        self.gamma = 0.5
        self.MDP = (self.S, self.A, self.P, self.R, self.gamma)

        # 策略1：随机策略
        self.Pi_1 = {
            "s1-保持s1": 0.5, "s1-前往s2": 0.5,
            "s2-前往s1": 0.5, "s2-前往s3": 0.5,
            "s3-前往s4": 0.5, "s3-前往s5": 0.5,
            "s4-前往s5": 0.25, "s4-概率前往s2": 0.25,
            "s4-概率前往s3": 0.25, "s4-概率前往s4": 0.25,
        }

        # 策略2
        self.Pi_2 = {
            "s1-保持s1": 0.6, "s1-前往s2": 0.4,
            "s2-前往s1": 0.3, "s2-前往s3": 0.7,
            "s3-前往s4": 0.5, "s3-前往s5": 0.5,
            "s4-前往s5": 0.3, "s4-概率前往s2": 0.2,
            "s4-概率前往s3": 0.25, "s4-概率前往s4": 0.25,
        }


# 仅仅是Sarsa算法，Q-learning算法以及Dyna-Q算法用到的env，不用于动态规划算法
class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前Agent位置的横坐标
        self.y = self.nrow - 1  # 记录当前Agent位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作，change[0]:上，change[1]:下,change[2]:左，change[3]:右，坐标系原点定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.x + change[action][1]))
        next_state = self.y + self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):  # 回归初始状态，坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y + self.ncol + self.x
