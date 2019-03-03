# encoding=utf-8

# KNN
# Fargo
# 2019.3.1

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# 1. 获取数据
digits = load_digits()
digits_data = digits.data
# print(digits_data.shape, '\n', digits_data)

# 2. 探索数据
print('第一幅图像：\n', digits.images[0])
print('第一幅图像代表的数字含义： ', digits.target[0])
print('第一幅图像显示：\n')
plt.gray()
plt.imshow(digits.images[0])
# plt.show()


# 3. 数据清洗
# 分割数据集， 30%的测试集， 70%的训练集
train_X,  test_X, train_y, test_y = train_test_split(digits_data, digits.target, test_size=0.3)

# 数据规范化
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

# 4 构建KNN模型
knn = KNeighborsClassifier()
knn.fit(train_X, train_y)
predict_y = knn.predict(test_X)

# 5 评估
score = accuracy_score(test_y,predict_y)
print(score)


















