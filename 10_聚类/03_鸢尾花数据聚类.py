import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")


datas = pd.read_csv('data/iris.data', sep=",", header=None)
# print(datas.head())
X = datas.iloc[:, :-1]

clf = KMeans(n_clusters=3,n_init='auto')
clf.fit(X)
print(clf.cluster_centers_)
print(clf.labels_)
pre = clf.predict(X.iloc[[-1],:])
print(pre)