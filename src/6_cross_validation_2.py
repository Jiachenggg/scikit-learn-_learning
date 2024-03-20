from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target
train_sizes, train_loss, test_loss = learning_curve(
    SVC(gamma=0.001), X, y, cv=10, scoring='neg_mean_squared_error',
    # SVC 中的 gamma 参数，它控制了核函数的影响范围。gamma 的值越大，每个样本点的影响范围就越小，决策边界会更加复杂，可能会导致过拟合；而 gamma 值越小，每个样本点的影响范围就越大，决策边界会更加平滑，可能会导致欠拟合
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1])  # 指定了在学习曲线中要评估的不同训练样本数量相对于整个数据集的比例

# 训练误差和交叉验证误差的均值
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

# 训练误差和交叉验证误差随着训练样本数量变化的趋势
plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
         label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
