from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# a = np.array([[10, 2.7, 3.6],
#                      [-100, 5, -2],
#                      [120, 20, 40]], dtype=np.float64)
# print(a)
# print(preprocessing.scale(a))


X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                           random_state=22, n_clusters_per_class=1, scale=100)
plt.scatter(X[:, 0], X[:, 1], c=y)  # y是类别标签，根据不同类别用不同颜色表示
plt.show()

X = preprocessing.scale(X)  # normalization step
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

clf = SVC()  # 创建一个支持向量机分类器SVC(Support Vector Classification)  在 scikit-learn 中，SVM(Support Vector Machine)可以用于分类（SVC）和回归（SVR）任务
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
