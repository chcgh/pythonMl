import numpy as np
import pandas as pd

# x1 = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 4])
# x2 = np.array(['S', 'M', 'S', 'L', 'S', 'S', 'L', 'L', 'L', 'S'])
# y = np.array([-1, 1, 1, 1, -1, -1, 1, 1, -1, 1])

x1 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 4])
x2 = np.array(['S', 'M', 'S', 'L', 'S', 'S', 'L', 'L', 'L', 'S'])
y = np.array([-1, 1, 1, 1, -1, -1, 1, -1, 1, 1])

train_D = np.vstack((x1,x2)).T
class MNB():
    def __init__(self,alpha=0):
        self.alpha = alpha
    def fit(self,X, Y):
        X_size = X.shape
        N = X.shape[0]
        n = X_size[1]
        N_yk = {}
        xi_list = []
        for i in range(n):
            xi_list.append(X[:, i])
        self.cls = np.unique(Y.reshape(-1))
        k = len(self.cls)
        for c in self.cls:
            N_yk['N_y=%s' % c] = len(X[y == c])

        N_yk_xi = {}
        for c in self.cls:
            for i, d in enumerate(xi_list):
                for e in np.unique(d):
                    N_yk_xi['Ny=%s,x%s=%s' % (c, i + 1, e)] = len(train_D[(y == c) & (d == e)])

        # 先验概率
        self.P_yk = {}
        for c in self.cls:
            self.P_yk['P_y=%s' % c] = (N_yk['N_y=%s' % c] + self.alpha) / (N + k * self.alpha)

        self.P_yk_xi = {}
        # 条件概率
        for c in self.cls:
            for i, d in enumerate(xi_list):
                for e in np.unique(d):
                    self.P_yk_xi['P_y=%s,x%s=%s' % (c, i + 1, e)] = (N_yk_xi['Ny=%s,x%s=%s' % (c, i + 1, e)] + self.alpha) / (
                                N_yk['N_y=%s' % c] + n * self.alpha)

    def predict(self,data):
        result = []
        self.predict_proba = []
        for x in data:
            pre = []
            for c in self.cls:
                p_c = self.P_yk['P_y=%s' % c]
                for i, e in enumerate(x):
                    p_c *= self.P_yk_xi['P_y=%s,x%s=%s' % (c, i + 1, e)]
                pre.append(p_c)
            self.predict_proba.append(pre)
            result.append(self.cls[pre.index(max(pre))])
        self.predict_proba = np.array(self.predict_proba)
        return result

test = np.array([[2,'M'],[3,'S']])
alpha = 1
algo = MNB(alpha)
algo.fit(train_D,y)
print('alpha=%s的预测结果：%s'%(alpha,algo.predict(test)))
print('各个类别的预测概率')
print(algo.predict_proba)


'''基于数组实现概率矩阵'''
class MultiNB():
    import numpy as np
    import pandas as pd
    def __init__(self,alpha=1):
        self.alpha = alpha

    def fit(self,X,Y):
        # class LenError(Exception): #定义一个异常类
        #     pass
        # if len(X) != len(Y):
        #     raise LenError('X与Y的长度不一致')

        assert len(X) == len(Y),'X与Y的长度不一致' #使用断言语句
        X = np.array(X)
        Y = np.array(Y).reshape(-1)
        x_size = X.shape
        N = x_size[0]
        self.n = x_size[1]
        self.Y_k = {i:k for i,k in enumerate(np.unique(Y))}
        self.k = len(self.Y_k)
        NB_P = []
        for c in self.Y_k.values():
            P_dict = {}
            c_datas = X[Y==c]
            N_yk = len(c_datas)
            P_dict['P(yk)'] = (N_yk+self.alpha)/(N+self.k*self.alpha)
            for i in range(self.n):
                x_i = c_datas[:,i]
                for x in np.unique(X[:,i]):
                    N_yk_xi = np.sum(x_i == x)
                    P_dict['P(x%s=%s|yk)'%(i+1,x)] = (N_yk_xi+self.alpha)/(N_yk+self.n*self.alpha)
            NB_P.append(pd.Series(P_dict))
        self.NB_P = pd.DataFrame(NB_P)
        self.NB_P.replace(np.nan,0,inplace=True)
        self.NB_P.index.name = 'yk'
        print(self.NB_P)

    def predict(self,X):
        result = []
        self.predict_proba = []
        for x in X:
            pre = []
            for c in range(self.k):
                P = self.NB_P.loc[c,'P(yk)']
                for i, e in enumerate(x):
                    P *= self.NB_P.loc[c,'P(x%s=%s|yk)'%(i+1,e)]
                pre.append(P)
            self.predict_proba.append(pre)
            result.append(self.Y_k[pre.index(max(pre))])
        result = np.array(result)
        try:
            result = result.astype(np.float)
        except:
            pass
        self.predict_proba = np.array(self.predict_proba)
        return result

#显示被省略的列
# pd.set_option('display.width',None)
# alog = MultiNB(alpha)
# alog.fit(train_D,y)
# print('alpha=%s的预测结果：%s'%(alpha,alog.predict(test)))
# print('各个类别的预测概率')
# print(alog.predict_proba)