import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

X0 = np.array([25, 28, 31, 35, 38, 40])  # 对温度数据增加一个维度并设置值为1
X = np.column_stack((np.ones((X0.shape[0], 1)), X0))
y = np.array([[106], [145], [167], [208], [233], [258]])
# x1 = X[:, 1]  # 取温度数据
# plt.scatter(x1.ravel(), y.ravel())  # ravel()函数返回以所有数据为元素的一个一维数组
# plt.show()

theta = np.zeros((2, 1))


def cost(theta):
    m = y.size
    y_hat = X.dot(theta)
    J = 1.0 / (2 * m) * np.square(y_hat - y).sum()
    return J


def gradientDescent_bgd(X, y, theta, alpha=0.01, iters=1500):
    m = y.size
    costs = []
    for i in range(iters):
        y_hat = X.dot(theta)
        theta -= alpha * (1.0 / m) * (X.T.dot(y_hat - y))
        c = cost(theta)
        costs.append(c)
    return costs


def gradientDescent2_bgd(X, y, theta, alpha=0.01, iters=1500):
    m = y.size
    costs = []
    for i in range(iters):
        sum_gradient = np.zeros(shape=theta.shape, dtype=float)

        for index in range(len(X)):
            y_pred = X[index:index + 1].dot(theta)
            sum_gradient += X[index:index + 1].T * (y_pred - y[index])

        theta -= alpha * (1.0 / m) * sum_gradient

        c = cost(theta)
        costs.append(c)
    return costs


def gradientDescent2_sgd(X, y, theta, alpha=0.01, iters=1500):
    costs = []
    for i in range(iters):
        for index in range(len(X)):
            y_pred = X[index:index + 1].dot(theta)
            gradient = X[index:index + 1].T * (y_pred - y[index])
            theta -= alpha * gradient
            c = cost(theta)
            costs.append(c)
    return costs


def gradientDescent2_mbgd(X, y, theta, alpha=0.001, iters=1500, batch_size=3):
    m = batch_size
    costs = []
    for i in range(iters):
        for j in range(0, len(X), batch_size):
            sum_gradient = np.zeros(shape=theta.shape, dtype=float)
            for k in range(batch_size):
                index = j + k
                y_pred = X[index:index + 1].dot(theta)
                sum_gradient += X[index:index + 1].T * (y_pred - y[index])

            theta -= alpha * (1.0 / m) * sum_gradient
            c = cost(theta)
            costs.append(c)
    return costs


def fsolve(X, y):
    X = np.mat(X)
    Y = np.mat(y)
    theta = (X.T * X).I * X.T * Y
    return theta.A


if __name__ == '__main__':
    costs=gradientDescent_bgd(X, y, theta=theta, alpha=0.001, iters=500000)
    # theta=fsolve(X, y)
    print("theta=", theta.ravel())
    print("cost=", cost(theta))



    costs=costs[1:]
    x=range(len(costs))
    plt.plot(x, costs)
    plt.show()

    x1 = X[:, 1]
    y_hat = X.dot(theta)
    plt.scatter(x1.ravel(), y.ravel())
    plt.plot(x1.ravel(), y_hat.ravel())
    plt.show()
