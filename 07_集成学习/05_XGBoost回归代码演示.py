import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# 导入数据集
data = pd.read_csv('./data/boston_housing.data', sep='\s+', header=None)

# 获取特征属性X和目标属性Y
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Xgboost训练过程
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, objective='reg:gamma')
# model = xgb.XGBRegressor(max_depth=5, learning_rate=0.2, n_estimators=100, objective='reg:squarederror')
model.fit(X_train, y_train)


print("训练集上的R^2 = {:.3f} ".format(r2_score(y_train, model.predict(X_train))))
print("测试集上的R^2 = {:.3f} ".format(r2_score(y_test, model.predict(X_test))))

# 绘制出特征重要性的图表，用于帮助我们了解哪些特征对于模型的预测效果影响较大，从而进行特征选择和优化。
# plot_importance(model)
# plt.show()
