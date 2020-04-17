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
        # edges are bi-directional
        # adding the edges and weights for both sides
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight

start_readDataIntoEdgeList = timer()
the_file = 'Data1.csv'
# read matrix without head.
# load the csv file using numpy library
a = np.loadtxt(the_file, delimiter=',', dtype=int)  # set the delimiter and he datatype
print(type(a)) # <class 'numpy.ndarray'>

print ('shape:',a.shape[0] ,"*", a.shape[1]) # shape: 8000 * 8000

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
# read the adjacency matrix file delimited by comma, and output the edgelist, for example

# adj matrix        edge list (from_node, to_node, weight=1)
# 0, 1, 0, 1        0,1,1
# 1, 0, 1, 0        0,3,1
# 0, 1, 0, 0 ->     1,2,1
# 1, 0, 0, 0
end_readDataIntoEdgeList = timer()
print("Read Data Into Edge List cost: " + str(end_readDataIntoEdgeList - start_readDataIntoEdgeList)) # Time in seconds, e.g. 5.38091952400282
print ('\nnum_edges:', num_edges) # num_edges: 8000, as same as the number of links Visone found
print ('')


# Depth First Search path finding function
def dfs_paths(graph, start, goal):
    # for counting the time we run the while loop
    count = 0
    # uses the stack data-structure to iteratively solve the problem,
    # stack list starting from the start
    # [starting node, [the path]]
    stack = [(start, [start])]
    while(stack):
        # set the limitation for the number of the paths we want (ed. only run 3 times while loop)
        if(count == 3):
            # get out the loop           
            break
        # get the vertex and the path from the stack
        (vertex, path) = stack.pop()
        # Using Python’s overloading of the subtraction operator to remove items from a set, to add only the unvisited adjacent vertices.
        for next in set(graph.edges[vertex]) - set(path):
            # if we find the end node
            if next == goal:
                # add 1 to the count
                count += 1
                # yielding each possible path when we locate the goal. Using a generator allows the user to only compute the desired amount of alternative paths
                # The first time the for calls the generator object created from your function, it will run the code in your function from the beginning until it 
                # hits yield, then it’ll return the first value of the loop. Then, each other call will run the loop you have written in the function one more time, 
                # and return the next value, until there is no value to return.
                yield path + [next]
            else:
                # keep finding the nodes
                stack.append((next, path + [next]))

# def bfs_paths(graph, start, goal):
#     count = 0
#     queue = [(start, [start])]
#     while queue:
#         if(count == 3):
#             break
#         (vertex, path) = queue.pop(0)
#         for next in set(graph.edges[vertex]) - set(path):
#             if next == goal:
#                 count += 1
#                 yield path + [next]
#             else:
#                 queue.append((next, path + [next]))


# Dijsktra algorithm function for finding the shortest path
def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous nodes, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    # Find nodes until it is the end node
    while current_node != end:
        # Record the node which are visited
        visited.add(current_node)
        # Find the dict of all possible next nodes for current node in graph
        destinations = graph.edges[current_node]
        # Record the weight to the current node from the tuple of (previous node, weight)
        weight_to_current_node = shortest_paths[current_node][1]

        # Loop through all possible next nodes
        for next_node in destinations:
            # Calculate the possible weight sum 
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            # Record the (current_node, weight) into the path if it doesn't find the next node already in the path
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            # if it already found the next node in the path
            else:
                # Refresh the current shortest weight as the total weight to the next node
                current_shortest_weight = shortest_paths[next_node][1]
                # if the current shortest weight is bigger then the possible weight sum
                if current_shortest_weight > weight:
                    # Record the (current_node, weight) into the path
                    shortest_paths[next_node] = (current_node, weight)
        # the potential next desitnations
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0] # previous nodes
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path, current_shortest_weight # (path, total weight cost for this path)


# Initialize the graph        
graph = Graph()
# Starting edge
start = 2837
# Ending edge
end = 7721
# start = 1
# end = 2

# adding edges from the edge list into the graph
for edge in edgeList:
    graph.add_edge(*edge)

# print(graph.edges)

# Find three different DFS paths
print("Depth_first_path: ")
start_dfs = timer()
dfs_list = list(dfs_paths(graph, start, end))
end_dfs = timer()
print("DFS search for 3 paths cost: " + str(end_dfs - start_dfs)) # Time in seconds, e.g. 5.38091952400282
# print("DFS path no.1: ", dfs_list[0]) # DFS path no.1: [2837, ..., 7721]
# print("\nDFS path no.2: ", dfs_list[1]) # DFS path no.2: [2837, ..., 7721]
# print("\nDFS path no.3: ", dfs_list[2]) # DFS path no.3: [2817, ..., 7721]
# Finding the nodes is the node which the other two paths don't have
for i in dfs_list[0]:
    if(i not in dfs_list[1]):
        print(i)
for i in dfs_list[0]:
    if(i not in dfs_list[2]):
        print(i)
for i in dfs_list[1]:
    if(i not in dfs_list[2]):
        print(i)

# Finding the number of nodes DFS went through
print("DFS Lenth 1")
print(len(dfs_list[0]))
print("DFS Lenth 2")
print(len(dfs_list[1]))
print("DFS Lenth 3")
print(len(dfs_list[2]))

start_dijsktra = timer()
# Find the shortest path 
shortest_path = dijsktra(graph, start, end)
end_dijsktra = timer()
print("Dijsktra search for shortest path cost: " + str(end_dijsktra - start_dijsktra)) # Time in seconds, e.g. 5.38091952400282
print("Shortest path: ")
print(shortest_path) # [(path , total weight cost for this path)]

