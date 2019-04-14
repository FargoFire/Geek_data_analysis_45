# encoding=utf-8

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# 1 读取数据
data = pd.read_csv('.\\26_kmeans-master\\data.csv', encoding='gbk')
train_X = data[['2019年国际排名', '2018世界杯', '2015亚洲杯']]


# 2 规范化数据
mm = MinMaxScaler()
train_X = mm.fit_transform(train_X)
print(train_X)


# 3 Kmeans 模型
kmeans = KMeans(n_clusters=3)
kmeans.fit(train_X)


# 4 聚类
predict_y = kmeans.predict(train_X)
print(predict_y)
data['等级'] = predict_y
print(data)