import sys

import numpy as np
import random

r = np.array([[-1, -1, -1, -1, 0, -1],
              [-1, -1, -1, 0, -1, 100],
              [-1, -1, -1, 0, -1, -1],
              [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, 100],
              [-1, 0, -1, -1, 0, 100]])
r = np.matrix(r)

q = np.zeros(r.shape)
q = np.matrix(q)

gamma = 0.8

max_state = r.shape[0] - 1  # 等于5

# 训练
for i in range(100):
    # 对于每一轮训练，随机选择一种状态
    state = random.randint(0, max_state)
    while state != max_state:
        # 选择r表中非负的值的动作
        r_pos_action = []
        for action in range(max_state + 1):
            if r[state, action] >= 0:
                r_pos_action.append(action)
        # 在这些动作中随机选择一个
        next_state = r_pos_action[random.randint(0, len(r_pos_action) - 1)]
        q[state, next_state] = r[state, next_state] + gamma * q[next_state].max()
        state = next_state

print(q.A)

# 预测
state = random.randint(0, max_state)
print('机器人处于{}'.format(state))
count = 0
while state != max_state:
    if count > 20:  # 如果尝试次数大于20次，则失败
        print('fail')
        break
    count += 1
    # 选择最大的q_max
    q_max = q[state].max()
    q_max_action = []
    for action in range(max_state + 1):
        if q[state, action] == q_max:  # 选择可行的下一个动作
            q_max_action.append(action)
    # 随机选择一个可行的动作
    next_state = q_max_action[random.randint(0, len(q_max_action) - 1)]
    print("机器人 goes to " + str(next_state) + '.')
    state = next_state
