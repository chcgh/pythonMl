import numpy as np
from sklearn.naive_bayes import BernoulliNB


x = np.array([[0, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 1],
              [1, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]])
y = np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0])


bnb = BernoulliNB()
bnb.fit(x, y)
day_pre = [[0, 0, 1, 0]]
pre = bnb.predict(day_pre)
print("预测结果如下\n:", '*' * 50)
print('结果为:', pre)
print('*' * 50)

# 进一步查看概率分布
pre_pro = bnb.predict_proba(day_pre)
print("不下雨的概率为：", pre_pro[0][0], "\n下雨的概率为：", pre_pro[0][1])
