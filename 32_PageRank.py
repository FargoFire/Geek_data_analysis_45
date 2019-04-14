#   encoding=utf-8

import networkx as nx
# 创建有向图
G = nx.DiGraph()
# 有向图之间边的关系
edges = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "A"), ("B", "D"), ("C", "A"), ("D", "B"), ("D", "C")]

for edge in edges:
    G.add_edge(edge[0], edge[1])

pagerank_list = nx.pagerank(G, alpha=0.85)
print(pagerank_list)

G.add_nodes_from(['X', 'Y', 'Z'])
G.add_edge('X', 'Y')
print(G.nodes(), G.number_of_nodes())
print(G.edges())

a = 'marshal,lcp@state.goy'
a = a.replace(',', ' ').split('@')[0]
print(a)
a = a.replace('a', '')
print(a)


