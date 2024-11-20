from collections import Counter 
from sklearn.datasets import fetch_openml
from imblearn.under_sampling import CondensedNearestNeighbour 
pima = fetch_openml('diabetes', version='1', parser='auto')
X, y = pima['data'], pima['target'] 
print('Original dataset shape %s' % Counter(y)) 
 
cnn = CondensedNearestNeighbour(random_state=42) 
X_res, y_res = cnn.fit_resample(X, y) 
print('Resampled dataset shape %s' % Counter(y_res)) 