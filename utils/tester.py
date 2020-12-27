import numpy as np

X, y = [1,2,3,4,5], [6,7,8,9]
X = np.array(X)
y = np.array(y)

idx = np.arange(X.shape[0])
np.random.shuffle(idx)

print(X[idx], y[idx])
