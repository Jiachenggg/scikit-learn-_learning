from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

clf = svm.SVC()
clf.fit(X, y)

# pickle 是 Python 内置的序列化库，而 joblib 则是 Scikit-learn 提供的用于大数据集的高效序列化库

# method 1: pickle
import pickle

# save
with open('../save/clf.pickle', 'wb') as f:  # 保存模型
    pickle.dump(clf, f)
# restore
with open('../save/clf.pickle', 'rb') as f:  # 加载保存的模型文件，将其存储在 clf2 变量中，并对新数据进行预测并打印结果
    clf2 = pickle.load(f)
    print(clf2.predict(X[0:1]))

# method 2: joblib
# import joblib
#
# # Save
# joblib.dump(clf, '../save/clf.pkl')
# # restore
# clf3 = joblib.load('../save/clf.pkl')
# print(clf3.predict(X[0:1]))
