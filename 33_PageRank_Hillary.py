# encoding=utf-8

# 通过PageRank 分析希拉里邮件中的重要人物关系

from collections import defaultdict
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 数据加载  读取邮件
emails = pd.read_csv('.\\33_PageRank-master\\Emails.csv')
# print(emails.columns)
emails_columns = ['Id', 'DocNumber', 'MetadataSubject', 'MetadataTo', 'MetadataFrom', 'SenderPersonId',
                  'MetadataDateSent', 'MetadataDateReleased', 'MetadataPdfLink', 'MetadataCaseNumber',
                  'MetadataDocumentClass', 'ExtractedSubject', 'ExtractedTo', 'ExtractedFrom', 'ExtractedCc',
                  'ExtractedDateSent', 'ExtractedCaseNumber', 'ExtractedDocNumber', 'ExtractedDateReleased',
                  'ExtractedReleaseInPartOrFull', 'ExtractedBodyText', 'RawText']
# 探究人物之间关系，只需处理 'MetadataTo', 'MetadataFrom' 数据

# 读取人物别名文件
Aliases_file = pd.read_csv('.\\33_PageRank-master\\Aliases.csv')
# print(Aliases_file.columns)  # ['Id', 'Alias', 'PersonId']
aliases = {}
for index, row in Aliases_file.iterrows():  # Aliases_names.iterrows()是一个生成器Generators
    aliases[row['Alias']] = row['PersonId']
# {'111th congress': 1, 'agna usemb kabul afghanistan': 2, 'ap': 3, 'asuncion': 4, 'alec': 5, ....}

# 读取人名文件
Persons_file = pd.read_csv('.\\33_PageRank-master\\Persons.csv')

persons = {}
for index, row in Persons_file.iterrows():
    persons[row['Id']] = row['Name']
# {1: '111th Congress', 2: 'AGNA USEMB Kabul Afghanistan', 3: 'AP', 4: 'ASUNCION', 5: 'Alec',....}


# 规范化名字
def unify_name(name):
    # 统一小写
    # print('原：', name)
    name = str(name).lower()
    # 去掉 ',' '_' 和 '@' 之后的邮箱
    name = name.replace(',', '').split('@')[0]
    name = name.replace(';', '')
    # 别名转换
    if name in aliases.keys():
        # print('转换后1：', persons[aliases[name]])
        return persons[aliases[name]]
    # print('转换后2：', name)

    return name


# 绘制人物网络图
def show_graph(graph, layout='spring_layout'):
    if layout == 'circular_layout':
        positions = nx.circular_layout(graph)
    else:
        positions = nx.spring_layout(graph)

    # 设置网络图中节点大小， 大小与pagerank 值相关，PageRank值小需 *20000
    node_size = [x['pagerank']*20000 for v, x in graph.nodes(data=True)]

    # 设置网络图中边长度
    edge_size = [np.sqrt(e[2]['weight']) for e in graph.edges(data=True)]

    # 绘制节点
    nx.draw_networkx_nodes(graph, positions, node_size=node_size, alpha=0.4)
    # 绘制边
    nx.draw_networkx_edges(graph, positions, edge_size=edge_size, alpha=0.2)
    # 绘制节点的label
    nx.draw_networkx_labels(graph, positions, font_size=10)
    # 输出邮件中的人物与希拉里的关系图
    plt.show()


# 将寄件人收件人名字规范化
emails.MetadataTo = emails.MetadataTo.apply(unify_name)
emails.MetadataFrom = emails.MetadataFrom.apply(unify_name)

# 设置边的权重等于发邮件的次数
edges_weight_temp = defaultdict(list)
for row in zip(emails.MetadataFrom, emails.MetadataTo, emails.RawText):

    temp = ( row[0], row[1] )
    if temp not in edges_weight_temp:
        edges_weight_temp[temp] = 1
    else:
        edges_weight_temp[temp] = edges_weight_temp[temp] + 1

# 转换格式
edges_weight = [ (key[0], key[1], val) for key, val in edges_weight_temp.items() ]

# 创建有向图
graph = nx.DiGraph()

graph.add_weighted_edges_from(edges_weight)

pagerank = nx.pagerank(graph)

nx.set_node_attributes(graph, name='pagerank', values=pagerank)

show_graph(graph)


# 精简人物关系图谱
# 设置PR的阈值， 筛选大于阈值的重要核心节点
pagerank_threshold = 0.003
# 复制一份完整的人物关系图
small_graph = graph.copy()
# 去掉PR值小于 设定阈值的节点
for n, p_rank in graph.nodes(data=True):
    if p_rank['pagerank'] < pagerank_threshold:
        small_graph.remove_node(n)

# 绘制新网络图，采用 circular_layout 将筛选的点围成一个圈
show_graph(small_graph, 'circular_layout')
show_graph(small_graph)





















