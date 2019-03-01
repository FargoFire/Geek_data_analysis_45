# encoding=utf-8

import pandas as pd
import numpy as np

# 1 获取数据集
train_data = pd.read_csv('.\\19_Titanic_Data-master\\train.csv')
test_data = pd.read_csv('.\\19_Titanic_Data-master\\test.csv')
gender = pd.read_csv('.\\19_Titanic_Data-master\\gender_submission.csv')

# 2 探索数据
# print(train_data.info()) # 了解数据表的基本情况：行数、列数、每列的数据类型、数据完整度
# print('-'*30)
# print(train_data.describe())
# print('-'*30)
# print(train_data.describe(include=['O']))  # 查看非数字的整体情况
# print('-'*30)


# 3清洗数据 ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
#  3.1‘PassengerId’，'name', 'Ticket' 对分类无用，  'Cabin' 缺失太多数据， 皆去除
del_labels = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train_data.drop(del_labels, axis=1, inplace=True)
test_data.drop(del_labels,axis=1, inplace=True)

# 3.2 Age Fare 数据缺失，可用平均值补齐
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# 3.3 Embarked为港口，可取众数
train_data['Embarked'].fillna(train_data['Embarked'].value_counts().index[0], inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].value_counts().index[0], inplace=True)

# 4 特征选择
features = [ 'Pclass',  'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]
print(train_data.info())

# 4.2 sex和Embarked的字符类型转换成数值型 Sex=male、Sex=female、Embarked=S、Embarked=C、Embarked=Q
from sklearn.feature_extraction import DictVectorizer
dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
test_features = dvec.fit_transform(test_features.to_dict(orient='record'))

# 5决策树模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 构造 ID3 决策树
clf = DecisionTreeClassifier(criterion='entropy')
# 决策树训练
clf.fit(train_features, train_labels)

# 6 模型预测与评估
# 6.1预测
pred_test = clf.predict(test_features)

# 6.2 评估
from sklearn.metrics import accuracy_score
test_labels = np.array( gender['Survived'] )
score = accuracy_score(test_labels, pred_test)
print(u'score 准确率为 %.4lf' % score)

# 得到决策树准确率
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'score 准确率为 %.4lf' % acc_decision_tree)





