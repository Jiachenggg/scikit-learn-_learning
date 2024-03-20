from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.fetch_california_housing()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4, :]))
print(model.coef_)
print(model.intercept_)
print(model.get_params())
print(model.score(data_X,
                  data_y))  # 计算了线性回归模型在整个数据集上的 R^2 决定系数（coefficient of determination）。R^2 值是衡量回归模型拟合优度的一种指标，其取值范围在0到1之间，越接近1表示模型拟合得越好
