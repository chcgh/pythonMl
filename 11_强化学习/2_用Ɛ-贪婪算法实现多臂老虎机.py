import numpy as np
import matplotlib.pyplot as plt

K = 5


# 伯努利老虎机：每个臂有概率p获得奖励1，概率1-p不能获得任何奖励
class BernoulliBandit:
    def __init__(self):
        self.probs = np.random.uniform(size=K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


def test_BernoulliBandit():
    np.random.seed(1)  # 设定随机数种子，使实验具有可重复性
    bandit = BernoulliBandit()

    for i in range(100):
        k = int(np.random.uniform(0, K))
        r = bandit.step(k)
        print("k=%d, r=%.4f" % (k, r))


class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(K)
        self.sumregret = 0
        self.actions = []
        self.regrets = []

    def update_regret(self, k):
        self.sumregret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.sumregret)

    # 选择第几号老虎机的策略，这里没有实现
    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


# 用epsilon-greedy策略实现run_one_step
class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆

        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class Plot:
    def plot_results(self, solver_list, solver_names):
        for idx, solver in enumerate(solver_list):
            time_list = range(len(solver.regrets))
            plt.plot(time_list, solver.regrets, label=solver_names[idx])
        plt.xlabel('Time steps')
        plt.ylabel('Cumulative regrets')
        plt.title('%d-armed bandit' % K)
        plt.legend()
        plt.show()


# 代码1：Ɛ-贪婪算法的累积懊悔几乎是线性增长的
class Main1(Plot):
    def __init__(self):
        np.random.seed(1)  # 设定随机数种子，使实验具有可重复性
        self.bandit = BernoulliBandit()
        self.epsilon_greedy_solver = EpsilonGreedy(self.bandit, epsilon=0.01)

    def run(self):
        self.epsilon_greedy_solver.run(5000)
        print('累积懊悔为：', self.epsilon_greedy_solver.sumregret)
        self.plot_results([self.epsilon_greedy_solver], ['EpsilonGreedy'])


# 代码2：无论Ɛ取值多少，累积懊悔都是线性增长的。随着Ɛ的增大，累积懊悔的速率也会增大。
class Main2(Plot):
    def __init__(self):
        np.random.seed(1)  # 设定随机数种子，使实验具有可重复性
        self.bandit = BernoulliBandit()
        self.epsilons = [1e-4, 0.04, 0.1, 0.25, 0.5]
        self.solver_list = [EpsilonGreedy(self.bandit, epsilon=e) for e in self.epsilons]
        self.solver_names = ['epsilon={}'.format(e) for e in self.epsilons]

    def run(self):
        for solver in self.solver_list:
            solver.run(5000)
        self.plot_results(self.solver_list, self.solver_names)


# epsilon值随时间衰减的epsilon-贪婪算法，具体衰减形式为反比例衰减，即第t时间步的epsilon值为1/t。
class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


# 代码3：随时间做反比例衰减的Ɛ-贪婪算法能够使累积懊悔与时间步的关系变成次线性(sublinear)的，这明显优于固定Ɛ值的Ɛ-贪婪算法
class Main3(Plot):
    def __init__(self):
        np.random.seed(1)  # 设定随机数种子，使实验具有可重复性
        self.bandit = BernoulliBandit()
        self.decay_epsilon_greedy_solver = DecayingEpsilonGreedy(self.bandit)

    def run(self):
        self.decay_epsilon_greedy_solver.run(5000)
        print('epsilon值衰减的贪婪算法的累积懊悔为：', self.decay_epsilon_greedy_solver.sumregret)
        self.plot_results([self.decay_epsilon_greedy_solver], ['DecayingEpsilonGreedy'])


if __name__ == '__main__':
    main = Main3()
    main.run()
    # test_BernoulliBandit()
