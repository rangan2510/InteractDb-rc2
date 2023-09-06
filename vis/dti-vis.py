#%%
import networkx as nx
G = nx.karate_club_graph()

# %%
for i in range(G.number_of_nodes()):
    print(G.nodes[i]['club'])
# %%
nx.get_node_attributes(G, "club")
# %%
from pyvis.network import Network
net = Network(notebook=True)
net.from_nx(G)
net.show("nx1.html")
# %%
nx_graph = nx.cycle_graph(10)
nx_graph.nodes[1]['title'] = 'Number 1'
nx_graph.nodes[1]['group'] = 1
nx_graph.nodes[3]['title'] = 'I belong to a different group!'
nx_graph.nodes[3]['group'] = 10
nx_graph.add_node(20, size=20, title='couple', group=2)
nx_graph.add_node(21, size=15, title='couple', group=2)
nx_graph.add_edge(20, 21, weight=5)
nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
nt =  Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True)
nt.toggle_physics(True)
# populates the nodes and edges data structures
nt.from_nx(nx_graph)
nt.show("nx2.html")
# %%
