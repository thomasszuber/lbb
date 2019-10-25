import networkx as nx 

import numpy as np 

import rome 

import json 

import pandas as pd

graph = rome.get_graph()

romes = list(graph.keys())

g = nx.Graph()
g.add_nodes_from(romes)
for rome1 in romes:
    current_edges = list(g.edges)
    for rome2 in romes:
        if (rome2,rome1) not in current_edges and (rome1 in graph[rome2] or rome2 in graph[rome1]):
            g.add_edge(rome1,rome2)


def get_distance(romes,g):
    distance = {}
    for r in romes:
        distance[r] = {} 
        for connex in (x for x in romes if x != r):
            try: 
                distance[r][connex] = nx.shortest_path_length(g,r,connex)
            except:
                distance[r][connex] = 20
        distance[r][r] = 0
    return distance

distance = get_distance(romes,g)

csv = {v:[] for v in ['rome','prev_rome','d']}
for prev_rome, romes in distance.items(): 
    for next_rome in romes: 
        csv['rome'].append(next_rome)
        csv['prev_rome'].append(prev_rome)
        csv['d'].append(distance[prev_rome][next_rome])

pd.DataFrame(csv).to_csv("../../bases/distances.csv")


isolated = [rome for rome in romes if np.mean([d for connex,d in distance[rome].items() if connex != rome]) >= 15]

romes = [rome for rome in romes if rome not in isolated]

distance = get_distance(romes,g)

file = open("distance.py", "w")

file.write('distance = ' + json.dumps(distance))
