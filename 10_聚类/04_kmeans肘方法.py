import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import silhouette_score  ###轮廓系数

import warnings

warnings.filterwarnings("ignore")

datas = pd.read_csv('data/iris.data', sep=",", header=None)
# print(datas.head())
X = datas.iloc[:, :-1]

sses = []
S = []
for K in range(2, 10):
    kmeans = KMeans(n_clusters=K, init="k-means++", n_init=10)
    kmeans.fit(X)
    inertia = kmeans.inertia_
    sses.append(inertia)
    # labels = kmeans.labels_
    # s = silhouette_score(X, labels)
    # print(s*100)
    # S.append(s)
plt.figure(num=1)
plt.plot(range(2, 10), sses)
# plt.figure(num=2)
# plt.plot(range(2, 10), S)
plt.show()
