# encoding=utf-8

# date：2019-02-28
# Fargo

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score


# 1.1 加载数据集
data = pd.read_csv('.\\23_breast_cancer_data-master\\data.csv')
# 1.2 数据探索
pd.set_option('display.max_columns', None)
# print(data.columns)
# print(data.describe())


# 2 数据清洗
# 2.1 将特征字段分成mean se worst 三组
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])

# 2.2 去除id
data.drop(['id'], axis=1, inplace=True)

# 2.3 将diagnosis的B(良性)和M(恶性) 换成0 1
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
# print(data['diagnosis'])

# 2.4 可视化 分析
# 肿瘤诊断结果
sns.countplot(x='diagnosis', data=data)
# plt.show()
# 热力图呈现 features_mean 字段之间的相关性
plt.figure(figsize=(20, 35))
plt.subplot(2, 3, 1)
corr = data[features_mean].corr()
sns.heatmap(corr, annot=True)    # annot 显示每个方格的数据，颜色浅相关性越大

plt.subplot(2, 3, 3)
corr = data[features_se].corr()
sns.heatmap(corr, annot=True)    # annot 显示每个方格的数据，颜色浅相关性越大

plt.subplot(2, 3, 5)
corr = data[features_worst].corr()
sns.heatmap(corr, annot=True)    # annot 显示每个方格的数据，颜色浅相关性越大
# plt.show()

# 3 特征选择
# 通过23_02.png 图可得出 mean se worst是对同一组内容的不同度量，选择保留mean一组特征，舍去se和worst

# 3.1 选择 10个特征 和全部特征
features_mean_all = ['radius_mean', 'texture_mean', 'perimeter_mean','area_mean',
                     'smoothness_mean', 'compactness_mean', 'concavity_mean',
                     'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

features_all = list(data.columns[1:31])

# 其中 'radius_mean', 'perimeter_mean','area_mean' 相关性大   选其一
# 'compactness_mean', 'concavity_mean','concave points_mean'相关性大   选其一
# 3.2 选择6个特征
features_remain = ['radius_mean', 'texture_mean', 'smoothness_mean','compactness_mean',
                   'symmetry_mean', 'fractal_dimension_mean']

def Train_test_create(data, features, size):
    # 3.3 准备训练集和测试集
    # 抽取30% 作为测试集
    train, test = train_test_split(data, test_size=size)
    # 抽取选定的特征的数值 作为训练集和测试集
    train_X = train[features]
    train_y = train['diagnosis']
    test_X = test[features]
    test_y = test['diagnosis']

    # 3.4 规范化数据
    # 采用Z-sore 规范化数据
    ss = StandardScaler()
    train_X = ss.fit_transform(train_X)
    test_X = ss.fit_transform(test_X)

    return train_X, train_y, test_X, test_y


def model_create(model, train_X, train_y, test_X, test_y):
    # 4 创建模型
    # 4.1 创建SVM 模型
    model = model
    print(model)
    # 训练集进行训练
    model.fit(train_X, train_y)
    # 测试集进行预测
    predicted = model.predict(test_X)

    # 5 评估
    score = accuracy_score(predicted, test_y)
    return score


train_X_6, train_y_6, test_X_6, test_y_6 = Train_test_create(data, features_remain, 0.3)
train_X_10, train_y_10, test_X_10, test_y_10 = Train_test_create(data, features_mean_all, 0.3)
train_X_all, train_y_all, test_X_all, test_y_all = Train_test_create(data, features_all, 0.3)

score_SVC_6 = model_create(SVC(kernel='rbf'), train_X_6, train_y_6, test_X_6, test_y_6)
score_LinearSVC_6 = model_create(LinearSVC(), train_X_6, train_y_6, test_X_6, test_y_6)
score_SVC_10 = model_create(SVC(), train_X_10, train_y_10, test_X_10, test_y_10)
score_LinearSVC_10 = model_create(LinearSVC(), train_X_10, train_y_10, test_X_10, test_y_10)
score_SVC_all = model_create(SVC(), train_X_all, train_y_all, test_X_all, test_y_all)
score_LinearSVC_all = model_create(LinearSVC(), train_X_all, train_y_all, test_X_all, test_y_all)

print('SVC_6 准确率：', score_SVC_6)
print('SVC_10 准确率：', score_SVC_10)
print('SVC_all 准确率：', score_SVC_all)
print('LinearSVC_6 准确率：', score_LinearSVC_6)
print('LinearSVC_10 准确率：', score_LinearSVC_10)
print('LinearSVC_all 准确率：', score_LinearSVC_all)





















