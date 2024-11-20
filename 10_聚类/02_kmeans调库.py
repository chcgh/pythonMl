import numpy as np #
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")


def demo1():
    X = np.array([[1, 2], [2, 2], [6, 8], [7, 8]])
    kmeans = KMeans(n_clusters=2, n_init='auto')
    kmeans.fit(X)
    print(kmeans.cluster_centers_)  ##簇中心
    print(kmeans.labels_)  ###簇的标签


def demo2():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_blobs  # 生成数据函数
    from sklearn import metrics

    #解决中文乱码
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    n_samples = 1500
    X, y = make_blobs(n_samples=n_samples, centers=4, random_state=170)
    X = StandardScaler().fit_transform(X)  # 标准化

    kmeans = KMeans(n_clusters=4, n_init='auto', random_state=170)
    kmeans.fit(X)

    # 反映同类样本类内平均距离尽可能小，类间距离尽可能大的指标。取值范围在[-1,1]之间，越大越好
    labels = kmeans.labels_
    pgjg1 = metrics.silhouette_score(X, labels, metric='euclidean')  # 轮廓系数，取值范围从-1到1
    print('聚类结果的轮廓系数=', pgjg1)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c='r')
    plt.title("聚类前数据图")
    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    plt.title("聚类后数据图")
    plt.show()


if __name__ == '__main__':
    # demo1()
    demo2()
