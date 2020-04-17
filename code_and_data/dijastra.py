# Shortest path: http://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/
# DFS path: https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
from collections import defaultdict
import numpy as np
from timeit import default_timer as timer

# the Graph class for mapping the edges
class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'2': ['5', '6', '9', '10'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('2', '6'): 2, ('2', '10'): 4, ...}
        """
        # using default dictionary to store the list of the edges
        self.edges = defaultdict(list)
        # the weights between nodes
        self.weights = {}

    # the function for adding the edges
    def add_edge(self, from_node, to_node, weight):
        # edges are bi-directionals
        # adding the edges and weights for both sides
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight

start_readDataIntoEdgeList = timer()
# the_file = 'test.csv'
the_file = 'Data1.csv'
# read matrix without head.
# load the csv file using numpy library
a = np.loadtxt(the_file, delimiter=',', dtype=int)  # set the delimiter and he datatype
print(type(a)) # <class 'numpy.ndarray'>

print ('shape:',a.shape[0] ,"*", a.shape[1]) # shape: 8000 * 8000

# num_nodes = a.shape[0] + a.shape[1] 

num_edges = 0

# the edge list store the connection 
edgeList = []

# Getting the connection
# double for loop untill 8000 * 8000
for row in range(a.shape[0]):
    for column in range(a.shape[1]):       
        # get rid of repeat edge
        # (column, row, weight=1)
        if (a.item(row,column) == 1 and (column,row, 1) not in edgeList): 
            num_edges += 1
            edgeList.append((row, column, 1))

# Initialize the graph        
graph = Graph()
# # Starting edge
# start = 2837
# # Ending edge
# end = 7721
start = 0
end = 1

import numpy as np
from collections import defaultdict


a = np.loadtxt('test.csv', delimiter=',', dtype=int)
# import the csv file

rows, cols = np.where(a == 1)
# Get the row and column coordinates where the array is 1
edges = zip(rows.tolist(), (cols).tolist())
# convert the rows and columns to list
# returns a zip object
edges = list(edges)
# change edges into list format


class Graph():
    def __init__(self):
        self.edges = defaultdict(list)
    # the self variable represents the instance of the object itself

    def add_edges(self, from_node, to_node):
        self.edges[from_node].append(to_node)
    # from start node to destination


graph = Graph()
# initialize the graph

for edge in edges:
    graph.add_edges(*edge)
# add all nodes (from_node, to_node) into edges list
print("get the data from csv file successfully")
# print(dict(graph.edges))
# print(edges)






from collections import defaultdict
from heapq import *

def dijkstra(edges, f, t):
    g = defaultdict(list)
    for l,r,c in edges:
        g[l].append((c,r))

    q, seen, mins = [(0,f,())], set(), {f: 0}
    while q:
        (cost,v1,path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t: return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 in seen: continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf")



print(dijkstra(edges_with_weight, 2837, 7721))