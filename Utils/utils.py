import torch
import pandas as pd
import numpy as np
import networkx as nx
class Utils():
	def read_data(self, path):
		rdata = []
		with open(path) as data:
			for line in data:
			    rdata.append(line)
		return rdata

	def split_raw_triple(self, raw_triple, ch = '\t'):
		''' splits the raw triple to get the triple (nodes, edges, nodes) '''
		triple = []
		for el in raw_triple:
			triple.append(el.split(ch))
		return triple

	def delete_edge(self, triple):
	    ''' deletes the edge labels from the triple and returns only the edges'''
	    for el in triple:
	        el.pop(1)
	    return triple

	def to_string_node(self, triple):
		''' converts the edges into strings '''
		for el in triple:
			for i in range(len(el)):
			    el[i] = str(el[i])
		return triple

	def drop_duplicate_node(self, edge):
		''' drops the duplicate edges'''
		n_edge = []
		for el in edge:
			if el not in n_edge:
				n_edge.append(el)  #dropping the duplicates
		return n_edge

	def nodemap(self, tup):
		''' enumerates every tuple'''
		key = {} # to store each encoding of a node
		i = 0
		for edge in tup:
			for j in range(len(edge)):
				if edge[j] in key: # if the node is already encoded then do nothing. If j is 1 implies edge[j] is an edge so we ignore
					continue
					
				else:
					key[edge[j]] = i # encode the node
					i += 1
		return key

	def mapback(self, key):
		''' maps back the node encoding to its node '''
		reverse_key = {}
		for node in key:
			reverse_key[key[node]] = node
		return reverse_key

	def preprocess(self, tup, key):
		''' returns the node encoding for each node '''
		for el in tup:
		    for j in range(len(el)):
		        el[j] = key[el[j]]  # changes each node by its encoding
		return tup

	def getadjlist(self, tup, key):
		'''returns the adjacency list of the graph where the index of the list are the nodes and elements w.r.t indexes are the adjacent nodes'''
		adj = [None] * len(key) # to store the adjacency list
		for node in range(len(adj)): 
		 	adj_v = [] #to store the set of nodes adjacent to each node
		 	for edges in triple:
		 		if edges[0] == node:
		 			adj_v.append(edges[1])
		 	if len(adj_v) != 0:
		 		adj[node] = adj_v
		return adj


	def getdegreedist(self, nodes):
		''' gets the outdegree distribution of the entire graph '''
		degreedist = {}
		i = 0		
		for node in nodes:
			if node[0] in degreedist:  
				degreedist[node[0]] += 1      # if node already in degree dictionary then increment its key value
			else:
				degreedist[node[0]] = 1
		return degreedist

	def getsubgraph(self, tup, nodes, adj = {}, hops = 1):
		''' returns a subgraph corresponding to the nodes specified and the hops specified'''
		sgraph = []
		if hops == 0:
			return []	
		for el in nodes:
			for edge in tup:
				if edge[0] == el:
					sgraph.append(edge)
		neighbours = []
		while(hops > 1):
			for node in nodes:
				if node in range(len(adj)):
					for nd in adj[node]:
						neighbours.append(nd)
						for edge in triple:
							if edge[0] == nd and edge not in sgraph:
								sgraph.append(edge)
			nodes = neighbours
			hops -= 1
		return sgraph


	def write_embeddings_tocsv(self, path,emb_df, index = False):
		'''accepts a dataframe as an input to convert them into a csv file'''
		emb_df.to_csv(path,index = index)
		print("CSV File Written successfully")


	def write_adjlist(self, path, adj):
		'''writes an adjacency list to a file'''
		with open(path, 'w') as f:
			for i in range(len(adj)):
				f.write(str(i) + " " + " ".join(map(adj[i], str)))
		f.close()
		print("File written successfully")


	def get_embeddingtensor(self, path):
		''' returns the embeddings as a tensor by reading the embedding csv file'''
		emb = pd.read_csv(path)
		emb_array = emb.to_numpy()
		return torch.tensor(emb_array)


	def loadGraphFromEdgeListTxt(self, file_name, directed=True):
	    with open(file_name, 'r') as f:
	        # n_nodes = f.readline()
	        # f.readline() # Discard the number of edges
	        if directed:
	            G = nx.DiGraph()
	        else:
	            G = nx.Graph()
	        for line in f:
	            edge = line.strip().split()
	            if len(edge) == 3:
	                w = float(edge[2])
	            else:
	                w = 1.0
	            G.add_edge(int(edge[0]), int(edge[1]), weight=w)
	    return G

	