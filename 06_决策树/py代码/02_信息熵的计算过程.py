# -- encoding:utf-8 --
"""
Create on 19/3/10
"""

import numpy as np


def entropy(p):
    return np.sum([-t * np.log2(t) for t in p if t != 0])
def gini(p):
    # return 1 - np.sum([t * t for t in p])
    return np.sum([t*(1-t) for t in p])

def h(p):
    return gini(p)
    # return entropy(p)

# print(entropy([0.6,0.4]))

#x1=是 p1=0.7   Y ： 5:2
#x1=否 p2=0.3   Y：1:2

# H(Y|x1) = p1*entropy([2/7,5/7]) + p2*entropy([1/3,2/3])

# print(0.7*entropy([2/7,5/7])+0.3*entropy([1/3,2/3]))

# print(entropy([0.25, 0.25, 0.25, 0.25]))
# print(entropy([0.65, 0.25, 0.05, 0.05]))
# print(entropy([0.97, 0.01, 0.01, 0.01]))
#
#
# print(entropy([0.5, 0.5]))
# print(entropy([0.7, 0.3]))
# print(entropy([0.4, 0.6]))
#
# print(entropy([0.5, 0.25, 0.25]))
# print(entropy([5.0 / 8, 3.0 / 8]))

# print(0.4 * entropy([0.25, 0.75]) + 0.6 * entropy([1 / 3, 2 / 3]))

print(h([0.0,1.0]))
print(h([0.6,0.4]))
print(h([0.4,0.4,0.2]))
print(h([0.2,0.2,0.2,0.2,0.2]))
print(h([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]))


