from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

# test train split #
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

# this is cross_val_score #
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=5)  # n_neighbors参数的选择会影响模型的性能和预测结果，通常需要通过交叉验证等方法来选择最优的K值
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print(scores)
# print(scores.mean())


# this is how to use cross_val_score to choose model and configs #
# 使用 cross_val_score 函数进行 K 折交叉验证，评估模型的性能，并输出每折交叉验证的准确率得分
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')  # for classification
    # loss = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error') # for regression

    k_scores.append(scores.mean())
    # k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
# plt.ylabel('Cross-Validated Loss')
plt.show()
