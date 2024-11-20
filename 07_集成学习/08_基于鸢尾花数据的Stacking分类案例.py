from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'simHei']

X, y = load_iris(return_X_y=True)
estimators = [
     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
     ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))
]
stacking = StackingClassifier(
     estimators=estimators, final_estimator=LogisticRegression()
)
softmax = LogisticRegression(C=0.1, solver='lbfgs', multi_class='multinomial', fit_intercept=False)
gbdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)
rf = RandomForestClassifier(max_depth=5, n_estimators=150)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

scores_train = []
scores_test = []
models = []
times = []

for clf, modelname in zip([softmax, gbdt, rf, stacking],
                          ['softmax', 'gbdt', 'rf', 'stacking']):
     print('start:%s' % (modelname))
     start = time.time()
     clf.fit(X_train, y_train).score(X_test, y_test)
     end = time.time()
     score_train = clf.score(X_train, y_train)
     score_test = clf.score(X_test, y_test)
     scores_train.append(score_train)
     scores_test.append(score_test)
     models.append(modelname)
     times.append(end - start)

print('scores_train:', scores_train)
print('scores_test', scores_test)
print('models:', models)
print('开始画图----------')

plt.figure(num=1)
plt.plot([0, 1, 2, 3], scores_train, 'r', label=u'训练集')
plt.plot([0, 1, 2, 3], scores_test, 'b', label=u'测试集')
plt.title(u'鸢尾花数据不同分类器准确率比较', fontsize=16)
plt.xticks([0, 1, 2, 3], models, rotation=0)
plt.legend(loc='lower left')

plt.figure(num=2)
plt.plot([0, 1, 2, 3], times)
plt.title(u'鸢尾花数据不同分类器训练时间比较', fontsize=16)
plt.xticks([0, 1, 2, 3], models, rotation=0)
plt.show()