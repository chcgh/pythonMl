# -- encoding:utf-8 --
"""
Create on 19/3/10
"""

import numpy as np


def entropy(p):
    return np.sum([-t * np.log2(t) for t in p if t != 0])


def h(p):
    return entropy(p)


print(h([0.0, 1.0]))
# print(h([0.6,0.4]))
print(h([0.5, 0.5]))
# print(h([0.4,0.4,0.2]))
# print(h([0.2,0.2,0.2,0.2,0.2]))
# print(h([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]))
# print(h([0.3,0.7]))
