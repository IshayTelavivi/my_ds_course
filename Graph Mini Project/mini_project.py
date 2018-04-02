"""
Python Final Mini Project‏
-------------------------
This project is concerned with the implementation of mathematical graphs and their usages. Project parts:
Part I - The Node class
Part II – The Graph class
Part III – Non-directional graph
"""


"""
Part I – The Node class
"""
# Task 1 – Define the class

import random
import datetime
import numpy as np

class Node():
    def __init__(self, name):
        self.name = name

        self.neighbors = {}  # This dictionary contains the node's neighbors. Empty upon node creation


    def __str__(self):
        print_line = 'Node name: {}, no. of neighbors: {}.'
        return print_line.format(self.name, len(self.neighbors))


    def __len__(self):
        return len(self.neighbors) # Returns the number of neighbors of self


    def __eq__(self, other):
        return self.name == other.name # Returns True if the two have the same name


    def __ne__(self, other):
        return self.name != other.name # Returns True if the two do not have the same name


    def is_neighbor(self, name):
        return name in self.neighbors # Returns True if the name is one of the neighbors (in the dictionary)


    def add_neighbor(self, name, weight=1):
        if name == self.name: # This method should not allow adding a neighbor with the same name as self
            print('Cannot be a neignbor of itself')
        elif self.is_neighbor(name): # This method should not allow adding a neighbor with a name of an existing neighbor
            print('{} is already a neighbor.'.format(name))
        else:
            self.neighbors[name] = weight # Adds the name and weight to the neighbors dictionary


    def remove_neighbor(self, name):
        self.neighbors.pop(name, None) # Adding the 'None' allow the line to run without an error if not a neighbor


    def get_weight(self, name):
        if self.is_neighbor(name):
            return self.neighbors[name] # Returns the weight of the relevant edge
        else:
            return None # This method returns None if name is not a neighbor of self


    def is_isolated(self):
        return len(self.neighbors) == 0  # Returns True if self has no neighbors


# Task 2 – Exemplary usage

# Question 1

node_list = []

def creating_nodes_and_print(num, name_intro, list_of_nodes):
    """
    The function creates a number of nodes, saves them in a list and prints them
    Input:
    num - the number of nodes
    name_intro - a string that opens each one of the node names
    list_of_nodes - a list to store all the new nodes created by the function
    The function returns nothing
    """

    for i in range(num):
        new_node = Node(name_intro + str(i)) # Create an instance of Node
        list_of_nodes.append(new_node) # Adds to the node list
        print(new_node)



# Question 2

def creating_graph(list_of_nodes):
    """
    The function creates a graph according to the following guidelines:
    - A node whose index in the list is larger than a given node by 1 is its neighbor (if exists)
    - A node whose index in the list is larger than a given node by 4 is its neighbor (if exists)
    - The last node in the list is neighbor of the first
    Input: list_of_nodes - the function receives a list of node instances
    The function returns nothing
    """

    for i in range(len(list_of_nodes)):
        if i+1 <= len(list_of_nodes)-1: # Making sure I don't get out of index range
            list_of_nodes[i].add_neighbor(list_of_nodes[i+1].name, random.randint(10, 30))
        if i+4 <= len(list_of_nodes)-1: # Making sure I don't get out of index range
            list_of_nodes[i].add_neighbor(list_of_nodes[i + 4].name, random.randint(10, 30))
    list_of_nodes[0].add_neighbor(list_of_nodes[-1].name, random.randint(10, 30)) # last node is neighbor of the first


# Now I am testing some things:
def check_graph(list_of_nodes):
    """
    The function makes a number of checks to test the implementation. For each node the function:
    - prints the node
    - prints its neighbors dictionary
    - prints total weight of the node
    - checks if node5 is a neighbor of the node
    - checks if the node is isolated
    Input: list_of_nodes - the function receives a list of node instances
    The function returs nothing
    """
    for node in list_of_nodes:
        print(node)
        print("neighbors: ", node.neighbors) # Print neighbors' list
        if list_of_nodes.index(node) != len(list_of_nodes)-1: # For all node except for the last one
            next_node = list_of_nodes[list_of_nodes.index(node)+1] # This is the node whose index is larger than
                                                                   # current node by 1
            print("Weight of {}:".format(next_node.name), node.get_weight(next_node.name)) # get_weight of next node
        print("Is node5 a neighbor:", node.is_neighbor("node5")) # Just a simple check of the is_neighbor function
        print("Isolated: ", node.is_isolated())
        print()

# Question 3 and 4

def analyze_edges_and_weight(list_of_nodes):
    """
    The function analyzes the data regarding the neighbors, edges and their weight for each node
    Input: list of nodes - the function receives a list of node instances
    The function return a string with the following:
    - total_n_edges: total number of edges in the graph
    - total_weight_of_graph: total weight of all edges in the graph
    - sorted_info: a list of tuples with the node names, the number of edges and their weight, sorted by the number
    of edges
    """
    edges_info = []
    for node in list_of_nodes:
        n_edge_of_node = len(node.neighbors) # Counts the kys in the dictionary 'Node.neighbors'
        total_weight_of_node = sum(list(map(lambda x: node.neighbors[x], node.neighbors))) # Sums values of the dict
        node_info = (node.name, n_edge_of_node, total_weight_of_node)
        edges_info.append(node_info)
    total_n_edges = sum([tup[1] for tup in edges_info]) # Sum total number of edges
    total_weight_of_graph = sum([tup[2] for tup in edges_info]) # Sum total weight of edges
    sorted_info = sorted(edges_info, key=lambda tup: tup[1], reverse=True)
    return "Total number of edges is {},\nTotal weight of the graph is {}:\nNodes sorted by no. of edges: {}.".format(total_n_edges, total_weight_of_graph, sorted_info)

print("Part I - The Node class")
# For clear print, I am calling the relevant function below the question number print
print("Exemplary usage - Question 1")
print("----------------------------")
creating_nodes_and_print(10, "node", node_list) # Calling the function

print("Exemplary usage - Question 2")
print("----------------------------")
creating_graph(node_list) # Calling the function, using the node_list as an argument
check_graph(node_list) # Calling the function, using the node_list as an argument

print("Exemplary usage - Question 3 and 4")
print("----------------------------------")
print(analyze_edges_and_weight(node_list))
print()



"""
Part II – The Graph class
"""

# Task 1 – Define the class

class Graph():
    def __init__(self, name, nodes=[]):
        self.name = name
        self.nodes = {}
        for node in nodes:
            self.nodes[node.name] = node # Keys are the names of the nodes, and values are the node instances
        self.message_not_in_self = "{} is not in {}" # A message I am using several times when node is not in graph

    def __str__(self):
        for name, node in self.nodes.items():
            print(node)
        return str()


    def __len__(self):
        return len(self.nodes) # Returns the number of nodes (within self.nodes)


    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.nodes.keys() # If key is a string, return True if it is a name in self.nodes (keys)
        elif isinstance(key, Node):
            return key in self.nodes.values() # If key is a Node, return True if it is a node in self.nodes (values)
        else:
            type_of_key = type(key) # If any other object type, print the following message
            print("{} does not include {} type objects in keys nor in values.".format(self.name, type_of_key))


    def __getitem__(self, name):
        if name in self: # Using the __contains__ method
            return self.nodes[name] # Get the dictionary value of name, which is the node iteself
        else:
            raise KeyError(self.message_not_in_self.format(name, self.name)) # Raise KeyError if name is not in the graph


    def add_node(self, node):
        if node not in self:
            self.nodes[node.name] = node # Adding the node to the self.nodes dict
        else: # If the name of the (input) node already exist in self.nodes
            self.nodes[node.name].neighbors.update(node.neighbors) # Updating the current node dict (within
                                                                   # self.nodes) with the input node's neighbors

    def __add__(self, other):
        new_graph = Graph(self.name + "+" + other.name) # Instantiating the new combined graph
        new_graph.nodes = self.nodes
        # I cannot just combine the two nodes dicts with the method 'update', because in case that there are nodes
        # with the same name, the neighbors of the node in other will overwrite the one in self.
        # So I am checking each one of the nodes in other.nodes if it exist in self.nodes (new_graph.nodes)
        for node_name, node_instance in other.nodes.items():
            if node_name not in new_graph: # Using the __contains__ method
                new_graph.nodes[node_name] = node_instance # If the dict does not include this node's name, add it
            else:
                new_graph.add_node(node_instance) # # If the dict includes this node's name, use the add_node method
        return new_graph


    def remove_node(self, name):
        self.nodes.pop(name, None) # Adding the 'None' allows the line to run without an error if name is not a node
        # If I remove a node, I also need to remove all the edges from other nodes to that node
        for other_node_name in self.nodes:
            if name in self.nodes[other_node_name].neighbors:
                self.remove_edge(other_node_name, name)

    def is_edge(self, frm_name, to_name):
        # If the frm_name is one of the nodes in the graph, return True if to_name is its neighbor.
        if frm_name in self:
            return to_name in self.nodes[frm_name].neighbors
        # If the frm_name is not one of the nodes in the graph, returns this message
        else:
            return self.message_not_in_self.format(frm_name, self.name)


    def add_edge(self, frm_name, to_name, weight=1):
        # Only if both nodes are in the graph
        if frm_name in self and to_name in self:
            # And if both nodes have different names, and to_name is not already a neighbor of frm_name
            if frm_name != to_name and to_name not in self.nodes[frm_name].neighbors:
                self.nodes[frm_name].add_neighbor(to_name) # Add to_name as a neighbor to frm_name. Use add_neighbor
                                                           # method of Node class
                self.nodes[frm_name].neighbors[to_name] = weight # Adding the new neighbor's weight
        # If any of the nodes are in the graph print the below message for it
        else:
            for node_name in [frm_name,to_name]:
                if node_name not in self:
                    print(self.message_not_in_self.format(node_name, self.name))


    def remove_edge(self, frm_name, to_name):
        # If the frm_name node is in the graph, use the 'remove_neighbor' method of Node class
        if frm_name in self:
            self.nodes[frm_name].remove_neighbor(to_name)
        # If the frm_name node is not in the graph, print the below message
        else:
            print(self.message_not_in_self.format(frm_name, self.name))


    def get_edge_weight(self, frm_name, to_name):
        # If the frm_name node is in the graph, use the 'get_weight' method of Node class
        if frm_name in self:
            return self.nodes[frm_name].get_weight(to_name) # Using the 'get_weight' method of Node class
        # If the frm_name node is not in the graph, print the below message
        else:
            return self.message_not_in_self.format(frm_name, self.name)


    def get_path_weight(self, path):
        # Based on the 'get_edge_weight', but refers to series of nodes
        weights = []
        for i in range(len(path)-1):
            weights.append(self.get_edge_weight(path[i], path[i+1]))
        if len(path) == 0:
            return None
        else:
            # if path is feasible (there are edges between the pairs), all(weights) is True
            if all(weights):
                path_weight = sum(weights)
                return path_weight
            else: # No feasible path
                return None

    def is_reachable(self, frm_name, to_name):
        if all ([frm_name in self,to_name in self]):
            queue = [frm_name] # This queue is where I keep the unvisited nodes to be tested
            visited = set() # This is where I keep all the node already tested. 'set' - for faster search
            while len(queue) > 0:
                # If the tested node is the to_name, we know it is reachable, and I can end the loop with True
                if self.nodes[queue[0]] == self.nodes[to_name]:
                    return True
                else:
                    if queue[0] not in visited: # Preventing repeatability. If already visited I already performed
                                                # the following
                        # Append all tested node's neighbors to the queue
                        for neighbor_name in self.nodes[queue[0]].neighbors.keys():
                            if neighbor_name not in visited:
                                queue.append(neighbor_name)
                        visited.add(queue[0]) # Add tested node to visited to prevent repeatability
                    queue.pop(0) # Need to take out the tested node, so eventually the while loop will end
            return False
        else:
            return "{} or {} or both are not nodes in {}".format(frm_name, to_name, self.name)


    def find_shortest_path(self, frm_name, to_name):
        path_queue = [[frm_name]] # path_queue stores all of the (partial) paths to be examined.
        legitimate_paths = [] # This is where legitimate paths will be stored
        # Part 1, the while loop, finds all the legitimate paths from frm_name to to_name
        while len(path_queue) > 0:
            # The idea is to extend the paths till I reach to_name (or till it is popped due to illegitimacy)
            temp_path = path_queue[0]
            last_node_in_temp = temp_path[-1] # Original temp_path[len(temp_path)-1]. I checked if I reached to_name
            if self.nodes[last_node_in_temp] == self.nodes[to_name]:
                legitimate_paths.append(temp_path) # If the last node is to_name, I add it to the legitimate_paths
                path_queue.pop(0) # I must take it out of the queue to allow the while loop to end
            else:
                # The for loop below extends the temp_path with all last node's neighbors
                for neighbor_name in self.nodes[last_node_in_temp].neighbors.keys():
                    if neighbor_name not in temp_path: # I don't want to create path with circles
                        new_path = []
                        if neighbor_name in self:
                            new_path = temp_path + [neighbor_name] # Extends the temp_path with the neighbor
                            path_queue.append(new_path) # Adds the new path to the queue
                        else:
                            print(self.message_not_in_self.format(neighbor_name, self.name))
                path_queue.pop(0) # I must take it out of the queue to allow the while loop to end
        # At this point I hve a list of legitimate paths
        # Part 2 - find the path with the lowest weight out of the legitimate paths
        if len(legitimate_paths) == 0:
            return None
        else:
            legitimate_paths_with_weight = []
            for path in legitimate_paths:
                path_weight = self.get_path_weight(path)  # Using self method get_path_weight
                paths_with_weight = (path, path_weight)
                legitimate_paths_with_weight.append(paths_with_weight)
            sorted_legitimate_path_with_weight = sorted(legitimate_paths_with_weight, key=lambda tup: tup[1])
            path_with_lowest_weight = sorted_legitimate_path_with_weight[0][0]
        return path_with_lowest_weight


# Task 2 – Exemplary usage
print("Part 2")
# First creating the first graph
node_list_A = []
creating_nodes_and_print(2, "A", node_list_A) # Creating 2 nodes with names starting with "A"
node_list_A[0].add_neighbor(node_list_A[1].name)
node_list_A[1].add_neighbor(node_list_A[0].name)
graphA = Graph("GraphA", node_list_A) # Creating an instance of Graph
# Creating the second graph
node_list_B = []
creating_nodes_and_print(3, "B", node_list_B) # Creating 3 nodes with names starting with "B"
node_list_B[0].add_neighbor(node_list_B[1].name)
node_list_B[1].add_neighbor(node_list_B[2].name)
node_list_B[2].add_neighbor(node_list_B[1].name)
graphB = Graph("GraphB", node_list_B) # Creating an instance of Graph
# Crating the third graph
node_list_C = []
creating_nodes_and_print(5, "C", node_list_C) # Creating 5 nodes with names starting with "C"
node_list_C[0].add_neighbor(node_list_C[1].name)
node_list_C[0].add_neighbor(node_list_C[2].name)
node_list_C[2].add_neighbor(node_list_C[3].name)
node_list_C[0].add_neighbor(node_list_C[4].name)
node_list_C[1].add_neighbor(node_list_C[4].name)
node_list_C[4].add_neighbor(node_list_C[0].name)
graphC = Graph("GraphC", node_list_C) # Creating an instance of Graph
print()

print("Exemplary usage - Question 1")
print("-----------------------------")
# Printing the graphs
print("GraphA")
print(graphA)
print()
print("GraphB")
print(graphB)
print()
print("GraphC")
print(graphC)
print()

# Combinig the graphs
comb_graph = graphA + graphB + graphC
print(comb_graph.name)
print(comb_graph) # Test __str__ method

# Question 2
print("Exemplary usage - Question 2")
print("-----------------------------")
# Testing methods
# Adding node D to the graph, adding an edge from C3, then removing the edge and removing D
# Testing the methods: add_node, remove_node, is_edge, add_edge, remove_edge, __getitem__, __contains
node_d = Node("D")
comb_graph.add_node(node_d)
print("Node D was added.")
print(comb_graph.nodes) # I want to see D as one of the nodes
comb_graph.add_edge("C3", "D", 20)
print("Neighbors of C3: ", comb_graph.nodes["C3"].neighbors) # Testing that D is a neighbor of C3
                                                             # Also testing the __getitem__ method
print('is_edge between C3 and D: ', comb_graph.is_edge("C3", "D"))
print('D in Graph4: ', "D" in comb_graph)
comb_graph.remove_edge("C3", "D")
comb_graph.remove_node("D")
# Now making sure that D is not linked to C3 and not part of the graph any more
print(comb_graph.nodes)
print("Neighbors of C3: ", comb_graph.nodes["C3"].neighbors)

#Now I am connecting the original graphs with edges:
comb_graph.add_edge("A1", "B1")
comb_graph.add_edge("B2", "C4")
comb_graph.add_edge("A0", "C2")

# Testing the methods get_path_weight, is_reachable, find_shortest_path
print("Path weight: ", comb_graph.get_path_weight(["B0", "B1", "B2", "C4", "C0"]))
print("Is A0 rechable from C0 (should be False): ", comb_graph.is_reachable("C0", "A0"))
print("Is A1 rechable from C2 (should be True): ", comb_graph.is_reachable("A1", "C2"))
print("Shortest path from A1 to C3: ", comb_graph.find_shortest_path("A1", "C3"))
print()

# Question 3

def sort_nodes_by_n_reachable(graph):
    """
    The function uses the is_reachable method and sorts all nodes in the graph by the number of reachable nodes
    input:
    - Graph instance
    The function returns a list of tuples, sorted by the number of reachable nodes
    """
    list_of_node_and_reachables_tups = [] # stores the number of reachable nodes per node
    # The following for loop finds the number of reachable nodes per node
    for node_to_test in graph.nodes:
        n_reachable = 0
        # The following for loop checks each node if it is reachable from node_to_test. If so, adds to the counter
        for node_is_reachable in graph.nodes:
            if graph.is_reachable(node_to_test, node_is_reachable) and node_to_test != node_is_reachable:
                n_reachable += 1
        # Adds a tuple with the node_to_test and the counter of reachable nodes
        list_of_node_and_reachables_tups.append((node_to_test, n_reachable))
    # At this point we have a list with tuples including the node name and its reachables. Now need to sort them
    sorted_nodes_by_reachable = sorted(list_of_node_and_reachables_tups, key=lambda tup: tup[1], reverse=True)
    return sorted_nodes_by_reachable

print("Exemplary usage - Question 3")
print("-----------------------------")
print("The following list present the nodes sorted by the number of reachable nodes")
print(sort_nodes_by_n_reachable(comb_graph))
print()

# Question 4
def pair_nodes_with_max_weight_in_shortest_path(graph):
    """
    The function takes each pair of nodes in the graph, checks its shortest path, and among all these paths takes
    the one with the highest weight
    input:
    - Graph instance
    The function returns a string with the pair of nodes with the highest weight on their shortest path
    """
    list_pairs_with_shortest_path = []
    # The following for loop takes each pair, and adds the weight of their shortest path to the list
    for frm_name in graph.nodes:
        for to_name in graph.nodes:
            if frm_name != to_name: # Irrelevant because the question asks "pair of nodes"
                shortest_path = graph.find_shortest_path(frm_name, to_name)
                if shortest_path != None: # We must ignore a pair with no path
                    path_weight = graph.get_path_weight(shortest_path)
                    if path_weight != None: # We must ignore path with None weight
                        pair_weight_tup = ((frm_name, to_name), path_weight)
                        list_pairs_with_shortest_path.append(pair_weight_tup)
    # print(list_pairs_with_shortest_path) # Was used to check the function
    sorted_list_pairs = sorted(list_pairs_with_shortest_path, key=lambda tup: tup[1], reverse=True)
    pair_with_max = sorted_list_pairs[0] # Pair with the highest weight
    return "Pair whose shortest path with max weight: {} to {}".format(pair_with_max[0][0], pair_with_max[0][1])

print("Exemplary usage - Question 4")
print("-----------------------------")
print(pair_nodes_with_max_weight_in_shortest_path(comb_graph))
print()

# Task 3 – The roadmap implementation
# Question 1
# 1 - First parse the files and get the region edges and their durations

VALID_NODES = ["North", "South", "West", "East", "Center"]
TEMPLATE_START_EW = '%d/%m/%Y %Hh%Mm'
TEMPLATE_END_EW = '%d/%m/%Y %Hh%Mm\n'
TEMPLATE_START_WE = '%I:%M:%S%p ; %b %m %y'
TEMPLATE_END_WE = '%I:%M:%S%p ; %b %m %y\n'

def get_nodes_and_weight_from_file(file, template_start, template_end):
    """
    This function parses a file with travels, finds the edges between regions and the mean duration per edge.
    The creates neighbors list per region, to be used for creating the Node and Graph classes
    Input
    - file: text/csv file with travel info in each line
    - template_start: the string representation format for the start detetime
    - template_end: the string representation format for the end detetime
    The function returns a dictionary, whose keys are the starting regions (nodes) and its values are the
    neighbors dictionaries per region
    """
    with open(file, "rt") as f:
        data = f.readlines()
        edge_list = [] # This list stores the tuples of valid start and end regions
        edge_duration_dict = {} # Per tuple as key, the dict stores durations per travel as value
        # The following for loop calculates the duration per travel and adds it to edge_duration_dict
        for line in data[1:]: # Starts from 1 since there is a header line
            split_line = line.split(",")
            # I found out the some lines have wrong hours figures, such as 25h, so I elimenate them
            try: # I found out the some lines have wrong hours figures, such as 25h, so I elimenate them
                start_time = datetime.datetime.strptime(split_line[1], template_start)
                end_time = datetime.datetime.strptime(split_line[3], template_end)
                travel_duration = end_time-start_time
            except:
                pass
            if split_line[0] in VALID_NODES and split_line[2] in VALID_NODES:
                nodes_frm_to =  (split_line[0], split_line[2]) # Defining the start and end regions as the edge
                # If the edge not in edge_list yet, append to the list, and add its duration to duration dict values
                if nodes_frm_to not in edge_list:
                    edge_list.append(nodes_frm_to)
                    edge_duration_dict[nodes_frm_to] = [travel_duration]
                # If the edge already in edge_list yet, just add its duration to duration dict values
                else:
                    edge_duration_dict[nodes_frm_to].append(travel_duration)
        # At this point we have pairs of regions with a list of durations per each
        edge_and_duration_dict = {} # This dictionary stores the mean duration per edge
        for edge, durations in edge_duration_dict.items():
            duration_array = np.array(durations) # Turning the duration list into an array for time saving
            edge_and_duration_dict[edge] = duration_array.mean()
        # At this point we have pairs of regions (from - to) with the mean travel duration for each pair
        # The rest of the code prepares the information to be aligned with Node and Graph definitions
        neighbors_by_region = {}
        start_regions = set([tup[0] for tup in edge_list]) # These are the 'from' regions of every edge
        # For each one of the 'from' regions, make neighbors dict(region and duration)
        # and stores in neighbors_by_region
        for region in start_regions:
            neighbors_dict = {}
            for edge, duration in edge_and_duration_dict.items():
                if edge[0] == region:
                    neighbors_dict[edge[1]] = duration
            neighbors_by_region[region] = neighbors_dict

        return neighbors_by_region

# Calling the fubction for the two files and assigning the output in the below variables
regions_and_neighbors_ew = get_nodes_and_weight_from_file("travelsew.csv", TEMPLATE_START_EW, TEMPLATE_END_EW)
regions_and_neighbors_we = get_nodes_and_weight_from_file("travelswe.csv", TEMPLATE_START_WE, TEMPLATE_END_WE)

# 2 - Creating the nodes and their neighbors

graph_ew_object_list = []
graph_we_object_list = []

def creating_nodes_for_regions(name_list, object_list, dict_of_neighbors_per_region):
    """
    The function creates a number of nodes and saves them in a list. Then assigns their neighbors
    Input:
    name_list - list of name of the Nodes (regions)
    object_list - a list to store all the new nodes created by the function
    dict_of_neighbors_per_region - a dictionary with the neighbors infor per region
    The function returns nothing
    """

    for name in name_list:
        new_node = Node(name) # Create an instance of Node
        object_list.append(new_node) # Adds to the object list
    # This for loop assigns to each Node its neighbors' dictionary
    for node in object_list:
        for region, neighbors  in dict_of_neighbors_per_region.items():
            if node.name == region:
                node.neighbors = dict_of_neighbors_per_region[region]

# Calling the function for both node sets
creating_nodes_for_regions(VALID_NODES, graph_ew_object_list, regions_and_neighbors_ew)
creating_nodes_for_regions(VALID_NODES, graph_we_object_list, regions_and_neighbors_we)


# 3 - creating the graphs

graph_ew = Graph("EW", graph_ew_object_list)
graph_we = Graph("WE", graph_we_object_list)

print("The roadmap implementation - Question1")
print("--------------------------------------")
# Print separate graphs for testing
print("WE graph")
print(graph_we)
print("EW graph")
print(graph_ew)

# 4 - adding the graphs
graph_travel = graph_ew + graph_we
print("Combined WE+EW graph")
print(graph_travel)

# Question 2
print("The roadmap implementation - Question2")
print("--------------------------------------")
# I could have used the information directly from the get_nodes_and_weight_from_file function, but I assumed that
# the idea is getting this information from the graph

all_edges_and_durations = [] # This list stores all the edged (frm_name, to_name) and the mean duration
for frm_name, node in graph_travel.nodes.items():
    for to_name, duration in graph_travel.nodes[frm_name].neighbors.items(): # Takes the neighbors from each node
        edge_and_duration = (frm_name, to_name, duration)
        all_edges_and_durations.append(edge_and_duration)
sorted_edges_and_durations = sorted(all_edges_and_durations, key=lambda tup: tup[2], reverse=True)
edge_longest_time = sorted_edges_and_durations[0] # Takes the edge with the longest duration
print("The longest time to travel takes {} from {} to {}.".format(edge_longest_time[2], edge_longest_time[0],\
                                                                 edge_longest_time[1]))

print()
"""
Part III - Non-directional graph
"""
print("Part III - Non-directional graph ")
# Task 1 – define the class

class NonDirectionalGraph(Graph):
    def __init__(self, name, nodes=[]):
        Graph.__init__(self, name, nodes=[])


    def remove_node(self, name):
        # If I remove a node, I also need to remove all the edges from other nodes to that node
        for neighbor_name in self.nodes[name].neighbors.keys():
            self.remove_edge(neighbor_name, name)
        self.nodes.pop(name, None) # Adding the 'None' allows the line to run without an error if name is not a node


    def add_edge(self, frm_name, to_name, weight_frm_to=1, weight_to_frm=1):
        # Only if both nodes are in the graph
        if frm_name in self and to_name in self:
            # And if both nodes have different names, and to_name is not already a neighbor of frm_name
            if frm_name != to_name and to_name not in self.nodes[frm_name].neighbors:
                self.nodes[frm_name].add_neighbor(to_name) # Add to_name as a neighbor to frm_name. Use add_neighbor
                                                           # method of Node class
                # In the NonDirectionalClass I allow two separate weights for both directions
                self.nodes[frm_name].neighbors[to_name] = weight_frm_to # Weight for first direction
                self.nodes[to_name].add_neighbor(frm_name) # Add to_name as a neighbor to frm_name. Use add_neighbor
                                                           # method of Node class
                self.nodes[to_name].neighbors[frm_name] = weight_to_frm  # Weight for second direction
        # If any of the nodes are in the graph print the below message for it
        else:
            for node_name in [frm_name,to_name]:
                if node_name not in self:
                    print(self.message_not_in_self.format(node_name, self.name))


    def remove_edge(self, frm_name, to_name):
        # Only if both nodes are in the graph
        if frm_name in self and to_name in self:
            self.nodes[frm_name].remove_neighbor(to_name)
            self.nodes[to_name].remove_neighbor(frm_name)
        # If any of the nodes are in the graph print the below message for it
        else:
            for node_name in [frm_name, to_name]:
                if node_name not in self:
                    print(self.message_not_in_self.format(node_name, self.name))

    # I assumed that A to B do not necessarily have the same weight as B to A. So I left the method get_edge_weight
    # as is, and the below method get_path_weight considers weight in both direction (A to B, B to A)

    def get_path_weight(self, path):
        # Based on the 'get_edge_weight', but refers to series of nodes
        weights = []
        for i in range(len(path)-1):
            # Counting the weight of the two directions
            weights.append(self.get_edge_weight(path[i], path[i+1]))
            weights.append(self.get_edge_weight(path[i+1], path[i]))
        if len(path) == 0:
            return None
        else:
            # if path is feasible (there are edges between adjacent pairs in the path), all(weights) is True
            if all(weights):
                path_weight = sum(weights)
                return path_weight
            else: # No feasible path
                return None

    # This method was added to answer question no. 3.
    # It is similar to find_shortest_path method, only with longest instead of shortest
    def find_longest_path(self, frm_name, to_name):
        path_queue = [[frm_name]] # path_queue stores all of the (partial) paths to be examined.
        legitimate_paths = [] # This is where legitimate paths will be stored
        # Part 1, the while loop, finds all the legitimate paths from frm_name to to_name
        while len(path_queue) > 0:
            # The idea is to extend the paths till I reach to_name (or till it is popped due to illegitimacy)
            temp_path = path_queue[0]
            last_node_in_temp = temp_path[-1] # Original temp_path[len(temp_path)-1]. I checked if I reached to_name
            if self.nodes[last_node_in_temp] == self.nodes[to_name]:
                legitimate_paths.append(temp_path) # If the last node is to_name, I add it to the legitimate_paths
                path_queue.pop(0) # I must take it out of the queue to allow the while loop to end
            else:
                # The for loop below extends the temp_path with all last node's neighbors
                for neighbor_name in self.nodes[last_node_in_temp].neighbors.keys():
                    if neighbor_name not in temp_path: # I don't want to create path with circles
                        if neighbor_name in self:
                            new_path = temp_path + [neighbor_name] # Extends the temp_path with the neighbor
                            path_queue.append(new_path) # Adds the new path to the queue
                        else:
                            print(self.message_not_in_self.format(neighbor_name, self.name))
                path_queue.pop(0) # I must take it out of the queue to allow the while loop to end
        # At this point I hve a list of legitimate paths
        # Part 2 - find the path with the lowest weight out of the legitimate paths
        if len(legitimate_paths) == 0:
            return None
        else:
            legitimate_paths_with_weight = []
            for path in legitimate_paths:
                path_weight = self.get_path_weight(path)  # Using self method get_path_weight
                paths_with_weight = (path, path_weight)
                legitimate_paths_with_weight.append(paths_with_weight)
            sorted_legitimate_path_with_weight = sorted(legitimate_paths_with_weight, key=lambda tup: tup[1],\
                                                        reverse=True) # Reverse since I am looking for the longest
            path_with_highest_weight = sorted_legitimate_path_with_weight[0][0]
        return path_with_highest_weight

# Task 2 – The social network implementation

def create_social_graph(file):
    """
    The function creates a non directional graph, based on a file the contains connections and disconnections
    between nodes that represent people.
    input: text file
    the function returns the following:
    - The non directional graph
    - The highest number of friendship simultaneously(int)
    - A dictionary with the highest number of connections per node (person)
    """
    social_graph = NonDirectionalGraph("SocialGraph")
    with open(file, "rt") as f:
        data = f.readlines()
        n_friendship = 0 # Represents the number of friendships in the graph in each iteration
        highest_n_friendship = 0 # Captures the highest record of n_friendship in the graph
        highest_n_neighbors_per_node_dict = {} # Captures the highest record of friendship per node
        for line in data:
            split_line = line.split()
            if "became" in split_line: # "became" is in lines where persons become connected
                for name in [split_line[0], split_line[2]]:
                    # The following if statement makes sure to instantiate the node and adds it to the graph
                    if name not in social_graph:
                        node = Node(name)
                        social_graph.add_node(node)
                        highest_n_neighbors_per_node_dict[name] = 0 ##
                social_graph.add_edge(split_line[0],split_line[2]) # Adds a connection between the nodes
                n_friendship += 1 # Updates the number of friendships
                # The following for loop updates the highest number of friends (neighbors) if it changes
                for name in [split_line[0], split_line[2]]:
                    if len(social_graph.nodes[name].neighbors) > highest_n_neighbors_per_node_dict[name]:
                        highest_n_neighbors_per_node_dict[name] = len(social_graph.nodes[name].neighbors)
            elif "cancelled" in split_line: # "became" is in lines where persons become disconnected
                social_graph.remove_edge(split_line[0], split_line[2])
                n_friendship -= 1 # Updates the number of friendships
            # In case any of the words "cancelled" or "became" is in the line
            else:
                print("Unrecognized line")
            # The following for loop updates the highest number of friendship if it changes
            if n_friendship > highest_n_friendship:
                highest_n_friendship = n_friendship
    return social_graph, highest_n_friendship, highest_n_neighbors_per_node_dict


social_graph = (create_social_graph("social.txt")[0])

# Question 1
print("The social network implementation  - Question1")
print("----------------------------------------------")
print("Highest number of simultaneous friendship was: {}.".format(create_social_graph("social.txt")[1]))
print()
# Question 2
print("The social network implementation  - Question2")
print("----------------------------------------------")
# The function was built in a way that not only Reuben can be tested but any person (node)
print("Maximum number of friends Reuben had simultaneously is {}.".format(create_social_graph("social.txt")[2]\
                                                                              ["Reuben"]))
print()
# Question 3
print("The social network implementation  - Question3")
print("----------------------------------------------")
# To solve this question I added a method called find_longest_path to the NonDirectionalGraph class
# This method is similar to the find_shortest_path, but takes the longer path instead.
# Assumption 1 - all the edges in the social_graph have the same weights.
# Under this assumption I allow myself to use the find_longest_past function (weight based) for the maximal path
# (length based)
# Assumption 2 - the graph that I am checking is the end point of the social_graph after all iterations
def find_maximal_path(graph):
    """
    The function takes a graph, and searches for the longest uncycled path, first between each pair of nodes,
    and then totally. The function does not check twicw the same pair (start-to-end and end-to-start) to save half
    of the checks
    input: graph: a Graph or NonDirectionalGraph instance
    The function returns a string with the maximal path and its length
    """
    maximal_path_in_graph = []  # This is where the maximal path in the graph is stored (updates)
    checked_pairs = set() # In order not to check a pair twice, the set stores the already checked pairs (and the
                          # reversed order of the tuple
    for node_start in graph.nodes.keys():
        for node_end in graph.nodes.keys():
            tup_of_nodes = (node_start, node_end) # This tuple is checked in checked_pairs
            # Need to make sure that the path is valid (!=Node) and that it is not checked yet
            if graph.find_longest_path(node_start,
                                       node_end) != None and tup_of_nodes not in checked_pairs:
                tup_of_nodes_reverse = (node_end, node_start) # This is a tuple with the opposite order
                checked_pairs.add(tup_of_nodes)
                checked_pairs.add(tup_of_nodes_reverse)
                maximal_path_for_pair = graph.find_longest_path(node_start, node_end)
                # Updates the maximal path per pair exceeds the current one
                if len(maximal_path_for_pair) > len(maximal_path_in_graph):
                    maximal_path_in_graph = maximal_path_for_pair
    return "Maximal path between nodes is: {}, and the length is {} nodes.".format(maximal_path_in_graph, \
                                                                                   len(maximal_path_in_graph))

# I commented the following line to prevent the long run. Can be uncommented.
#print(find_maximal_path(social_graph))

# Question 4
print("The social network implementation  - Question4")
print("----------------------------------------------")

def suggest_friend(graph, node_name):
    """
    This function recommends a friend for node_name, based on the highest common friends between them. The suggested
    friend cannot be one of his connections
    input:
    - graph: a Graph or NonDirectionalGraph instance
    - node_name: a name of a noe in the graph, whom we are suggesting the friend
    The function returns a string with suggested friend and the number of common friends
    """
    suggested_friends = [] # This is where all the potential suggested friends (from second degree) are stored
    checked_node = set() # Here are already checked second neighbors, not to repeat the check
    for first_neighbor_name in graph.nodes[node_name].neighbors.keys():
        for second_neighbor_name in graph.nodes[first_neighbor_name].neighbors.keys():
            # I am not checking automatically every neighbor of second degree, it need to meet the folloewing
            # conditions: It cannot be already checked node, it cannot be node_name, and not any of its neighbors
            if all([second_neighbor_name not in checked_node, second_neighbor_name != node_name,\
                    second_neighbor_name not in graph.nodes[node_name].neighbors.keys()]):
                common_friends = 0
                # The following for loop check the common friend of second_neighbor and node_name. It actually
                # chacks if the neighbors of third degree are also the first neighbors (node_name connections)
                for third_neighbors_name in graph.nodes[second_neighbor_name].neighbors.keys():
                    if third_neighbors_name in graph.nodes[node_name].neighbors.keys():
                        common_friends +=1
                friend_potential = (second_neighbor_name, common_friends) # A summary of the potential of second
                suggested_friends.append(friend_potential)
                checked_node.add(second_neighbor_name)
    sorted_suggested_friends = sorted(suggested_friends, key=lambda tup: tup[1], reverse = True)
    suggest_friend_with_most_common = sorted_suggested_friends[0][0]
    n_common_friends_for_suggested = sorted_suggested_friends[0][1]
    return "The suggested friend for {0} is {1}, with {2} commmon friends.".format(node_name, suggest_friend_with_most_common, n_common_friends_for_suggested)

print(suggest_friend(social_graph, "Issachar"))