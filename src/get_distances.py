# %%
import networkx as nx 

import numpy as np 

import rome 

import json 

import pandas as pd

import itertools

graph = rome.get_graph()

romes = list(graph.keys())

#g = nx.Graph()
#g.add_nodes_from(romes)
#for rome1 in romes:
#    current_edges = list(g.edges)
#    for rome2 in romes:
#        if (rome2,rome1) not in current_edges and (rome1 in graph[rome2] or rome2 in graph[rome1]):
#            g.add_edge(rome1,rome2)
            
g = nx.DiGraph(graph)


def get_distance(romes,g):
    distance = {}
    d_max = 0 
    for r in romes:
        distance[r] = {} 
        for connex in (x for x in romes if x != r):
            try: 
                d = nx.shortest_path_length(g,r,connex)
                distance[r][connex] = d
                if d > d_max:
                    d_max = d 
            except:
                distance[r][connex] = np.inf
        distance[r][r] = 0
    for r,connex_romes in distance.items():
        for connex in connex_romes:
            if distance[r][connex] == np.inf:
                distance[r][connex] = d_max + 1
             
    return distance, d_max+1

distance, default = get_distance(romes,g)

# %%
distrib = {rome:{k: len(list(v)) for k, v in itertools.groupby(sorted(x.values()))} for rome,x in distance.items()} 
nb_inf = {rome:x[np.inf] for rome,x in distrib.items()}
# %%

csv = {v:[] for v in ['rome','prev_rome','d']}
for prev_rome, romes in distance.items(): 
    for next_rome in romes: 
        csv['rome'].append(next_rome)
        csv['prev_rome'].append(prev_rome)
        csv['d'].append(distance[prev_rome][next_rome])

pd.DataFrame(csv).to_csv("../../bases/distances.csv")

# %% 
isolated = [rome for rome in romes if np.min([d for connex,d in distance[rome].items() if connex != rome]) == default]

# %%
pd.DataFrame({'rome':isolated}).to_csv("../../bases/isolated_directed.csv")

# %%
romes = [rome for rome in romes if rome not in isolated]

distance, default = get_distance(romes,g)

file = open("distance.py", "w")

file.write('distance = ' + json.dumps(distance))
