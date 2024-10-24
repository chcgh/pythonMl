import numpy as np


class Linear:
    def __init__(self, use_b=True):
        self.use_b = use_b
        self.theta = None
        self.theta0 = 0

    def train(self, X, Y):
        if self.use_b:
            X = np.column_stack((np.ones((X.shape[0], 1)), X))
        # 二、为了求解比较方便，将numpy的'numpy.ndarray'的数据类型转换为矩阵的形式的。
        X = np.mat(X)
        Y = np.mat(Y)
        # 三、根据解析式的公式求解theta的值
        theta = (X.T * X).I * X.T * Y
        if self.use_b:
            self.theta0 = theta[0]
            self.theta = theta[1:]
        else:
            self.theta0 = 0
            self.theta = theta

    def predict(self, X):
        predict_y = X * self.theta + self.theta0
        return predict_y

    def score(self, X, Y):
        # mse

        # r^2

        # mae

        pass

    def save(self):
        """

        :return:
        """
        # self.theta0
        # self.theta

    def load(self, model_path):
        # self.theta0
        # self.theta

        pass


if __name__ == '__main__':
    X1 = np.array([10, 15, 20, 3, 50, 60, 60, 70]).reshape((-1,
                                                            1))  # numpy.reshape(-1,1)是reshape函数的一个特殊用法，其中-1是一个占位符，表示该维度的大小由Numpy自动计算，以保证总元素数不变。而1则指示我们想要将数据重塑为一个列向量。简单来说，numpy.reshape(-1,1)的作用是将一个数组转换成一个二维数组，其中只有一列，行数则是由原数组的总元素数决定。
    Y = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 30.5]).reshape((-1, 1))
    linear = Linear()
    linear.train(X1, Y)
    x_test = [[55]]
    y_test_hat = linear.predict(x_test)
    print(y_test_hat)
    print(linear.theta)
    print(linear.theta0)
