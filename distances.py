## DISTANCES 

graph = rome.get_graph()
romes = list(graph.keys())

g = nx.Graph()
g.add_nodes_from(romes)
for rome1 in romes:
    current_edges = list(g.edges)
    for rome2 in romes:
        if (rome2,rome1) not in current_edges and (rome1 in graph[rome2] or rome2 in graph[rome1]):
            g.add_edge(rome1,rome2)
                