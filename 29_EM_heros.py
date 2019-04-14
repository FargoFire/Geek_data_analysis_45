# encoding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


# 1 获取数据
data_ori = pd.read_csv('.\\29_EM_master\\heros.csv', encoding='gb18030')
features = data_ori.columns[:-2]
data = data_ori[features]
# ['英雄', '最大生命', '生命成长', '初始生命', '最大法力', '法力成长', '初始法力', '最高物攻', '物攻成长',
# '初始物攻', '最大物防', '物防成长', '初始物防', '最大每5秒回血', '每5秒回血成长', '初始每5秒回血',
# '最大每5秒回蓝', '每5秒回蓝成长', '初始每5秒回蓝', '最大攻速', '攻击范围']

# 2 探索数据-可视化
plt.rcParams['font.sans-serif'] = ['SimHei']          # 显示中文
plt.rcParams['axes.unicode_minus'] = False            # 显示负号


# '最大攻速', '攻击范围' 转化为数字量
data[u'攻击范围'] = data[u'攻击范围'].map({'远程': 1, '近战': 0})
data['最大攻速'] = data['最大攻速'].apply(lambda x: float(x.strip('%'))/100)
# print(data)


# 热力图显示
corr = data[features].corr()  # 相关性
# print(corr)
plt.figure(figsize=(12, 12))
sns.heatmap(corr, annot=True)
# plt.show()

# 根据相关性 将相关性大于0.8的特征降维
# ['最大生命','生命成长', '最大物防', '物防成长', '最大每5秒回血', '每5秒回血成长', '初始每5秒回血']
# ['最大法力', '法力成长', '初始法力', '最大每5秒回蓝', '每5秒回蓝成长']
# ['最高物攻', '物攻成长']
features_s = ['最大生命', '初始生命', '最大法力', '最高物攻', '物攻成长', '初始物攻', '最大物防',
              '初始物防', '最大每5秒回血', '最大每5秒回蓝', '初始每5秒回蓝', '最大攻速', '攻击范围']

features_all = ['最大生命', '生命成长', '初始生命', '最大法力', '法力成长', '初始法力', '最高物攻', '物攻成长',
                '初始物攻', '最大物防', '物防成长', '初始物防', '最大每5秒回血', '每5秒回血成长', '初始每5秒回血',
                '最大每5秒回蓝', '每5秒回蓝成长', '初始每5秒回蓝', '最大攻速', '攻击范围']
data_s = data[features_all]

ss = StandardScaler()
data_s = ss.fit_transform(data_s)

gmm = GaussianMixture(n_components=5, covariance_type='full')
gmm.fit(data_s)
predict_s = gmm.predict(data_s)

print(predict_s)

from sklearn.metrics import calinski_harabaz_score
print(calinski_harabaz_score(data_s, predict_s))










