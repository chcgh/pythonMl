import pandas as pd
import numpy as np

# TODO: 自己将下面的这些代码结合决策树的执行过程，整理成使用Python实现分类的决策树算法。
df = pd.read_csv('../datas/dt.data', header=None, names=['x1', 'x2', 'x3', 'y'], sep=',')
df.info()
# 计算一下df数据中，y=否和y=是的概率
# 获取y值为"是"的数据，从而计算概率值
c1 = df['y'] == '是'
tmp_df = df[c1]
p1 = len(tmp_df) / len(df)

# 获取y值为"否"的数据，从而计算概率值
c1 = df['y'] == '否'
tmp_df = df[c1]
p2 = len(tmp_df) / len(df)
print((p1, p2))

# 保存y=否和y=是的概率
p_y = []
df_y = df['y']
total_samples = len(df_y)
for y_v in np.unique(df_y.values):
    c = df_y == y_v
    tmp_df = df[c]
    p = len(tmp_df) / total_samples
    p_y.append(p)
print(p_y)
print("信息熵:{}".format(p_y))

# 针对x1计算信息熵
df_x1 = df['x1']
p_x1_y = []
for x1_v in np.unique(df_x1.values):
    # 获取当x1的特征值为x1_v的时候对应的数据
    df_x1_v = df[df_x1 == x1_v]
    # 获取x1的特征值为x1_v中的数据的标签y值
    df_x1_v_y = df_x1_v['y']
    # 统计当前x1对应的标签y值中，各个y的取值的概率
    p_y = []
    total_samples_x1_v = len(df_x1_v_y)
    print(total_samples_x1_v / total_samples)
    for y_v in np.unique(df_x1_v_y.values):
        c = df_x1_v_y == y_v
        tmp_df = df_x1_v_y[c]
        p = len(tmp_df) / total_samples_x1_v
        p_y.append(p)
    p_x1_y.append(p_y)
print(p_x1_y)

# 下面的代码看看即可
arr = ['否', '否', '否', '否', '否', '是', '是', '否', '是', '否']
true_count = 0
false_count = 0
for item in arr:
    if item == '是':
        true_count += 1
    else:
        false_count += 1
print([true_count / (true_count + false_count), false_count / (false_count + true_count)])

# TODO: 如果觉得比较难，先考虑特征属性只有离散值的情况
