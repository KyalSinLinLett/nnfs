import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

from knn import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = KNN(k=4)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(predictions)
print(y_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)


# print(X_train.shape)
# print(X_train[:5])
# print(X_test[:5])

# print([x for x in X_test])
# print([x for x in X_train])

# print(y_train.shape)
# print(y_train[:5])

# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], cmap=cmap, edgecolors='k', s=20)
# plt.show()
