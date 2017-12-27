# graphrep.py

import networkx as nx
import random
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
import numpy as np



# Regular tree that spreads messages according to Bitcoin protocol
class RegularTree(nx.Graph):

	def __init__(self, degree = None, spreading_time = None):
		super(RegularTree, self).__init__()
		self.tree_degree = degree
		self.source = 0
		self.max_node = 0 # highest-index node in the list 
		self.active = [0] # list of nodes that are not fully surrounded by infected nodes
		self.spreading_time = spreading_time
		if self.spreading_time is None:
			self.spreading_time = (self.tree_degree * 2 + 1)

		RegularTree.add_node(self,self.source, infected = True)

	def draw_plot(self):
		values = ['r' for i in self.nodes()]
		# values[-1] = 'b'

		pos=nx.circular_layout(self) # positions for all nodes

		nx.draw(self, pos = pos, node_color = values)

		labels={}
		for i in self.nodes():
			labels[i] = str(i)
		nx.draw_networkx_labels(self,pos,labels,font_size=16)

		plt.show()


	def add_node(self, u, attr_dict = None, **attr):
		super(RegularTree, self).add_node(u, attr)
		self.max_node = max(self.nodes())

	def get_neighbors(self, sources):
		neighbors = []
		for source in sources:
			neighbors += self.neighbors(source)
		return list(set(neighbors))

	def subgraph(self, nbunch):
		H = nx.Graph.subgraph(self, nbunch)
		return H

	def add_edges(self, source, node_list):
		for node in node_list:
			self.add_node(node, infected = False)
			self.add_edge(source, node)

	def infect_node(self, source, target):
		if (target == self.adversary):
			self.remove_edge(source, target)
		else:
			self.node[target]['infected'] = True
			self.active += [target]
				

		# Check if the source still has active neighbors; if not, remove it from active list
		uninfected_neighbors = self.get_uninfected_neighbors(source)
		if not uninfected_neighbors:
			self.active.remove(source)

	def get_uninfected_neighbors(self, source):
		neighbors = self.neighbors(source)
		uninfected_neighbors = [neighbor for neighbor in neighbors if self.node[neighbor]['infected'] == False]
		return uninfected_neighbors


class RegularTreeDiffusion(RegularTree):

	def __init__(self, degree = None, spreading_time = None, theta = 1):
		''' NB: Here the spreading_time	is actually the number of rings of the graph to infect'''
		super(RegularTreeDiffusion, self).__init__(degree, spreading_time)
		self.lambda1 = 1 # spreading rate over the diffusion graph
		self.theta = theta # spreading rate from a node to the adversary
		self.adversary_timestamps = {} 		# dictionary of adversary report time indexed by node
		self.received_timestamps = {}		# dictionary of message receipt time indexed by node
		self.received_timestamps[self.source] = 0

		
	def spread_message(self, first_spy_only = False):
		'''first_spy_only denotes whether this diffusion spread will only be used
		to measure the first spy adversary. In that case, some time-saving optimizations
		can be implemented. Most of the time, this flag will be set to false.'''
		
		count = 0
		stopping_time = self.spreading_time
		while self.active and self.max_node < 1500:
			# count += 1
			new_boundary = []
			# cycle through the active nodes, and spread with an exponential clock
			for node in self.active:
				# Check that all the nodes have enough neighbors
				if self.degree(node) < (self.tree_degree):
					# num_missing = 0
					# if node == 0 and self.degree(node) < self.tree_degree - 2:
					# 	num_missing = (self.tree_degree - 2) - self.degree(node)
					# elif not node == 0:
					num_missing = (self.tree_degree) - self.degree(node)

					# if num_missing > 0:
					new_nodes = range(self.max_node + 1, self.max_node + num_missing + 1)
					self.add_edges(node, new_nodes)

				# Adversary infection time
				self.adversary_timestamps[node] = self.send_to_adversary(node) 

				if first_spy_only and (node == self.source):
					stopping_time = min(stopping_time, self.adversary_timestamps[node])
					# print 'stopping_time', stopping_time


				# Neighbor infection times
				for new_node in new_nodes:
					rx_timestamp = self.send_to_neighbor(node)
					self.received_timestamps[new_node] = rx_timestamp
					# Add the new nodes to the boundary, but only if they are before cutoff time
					if rx_timestamp < stopping_time:
						new_boundary += [new_node]
			self.active = [i for i in new_boundary]

	def send_to_adversary(self, node):
		return self.received_timestamps[node] + np.random.exponential(1.0 / self.theta)

	def send_to_neighbor(self, node):
		return self.received_timestamps[node] + np.random.exponential(1.0 / self.lambda1)



class RegularTreeTrickle(RegularTree):

	def __init__(self, degree = None, spreading_time = None, theta = 1):
		''' Runs trickle spreading on a regular tree
			Args:
			spreading_time 		number of timesteps to run 
			theta				number of connections to the eavesdropper per node
		'''
		super(RegularTreeTrickle, self).__init__(degree, spreading_time)
		# self.adversary = -1
		# self.adversary_timestamps = SortedDict()
		# self.add_node(self.adversary, infected = False)
		self.source = 0
		self.theta = theta # spreading rate from a node to the adversary
		# self.adversary_timestamps = {} 		# dictionary of adversary report time indexed by node
		# self.received_timestamps = {}		# dictionary of message receipt time indexed by node
		# self.received_timestamps[self.source] = 0


		# def add_edge(self, u, v):
		# 	super(RegularTree, self).add_edge(u, v)
		# 	super(RegularTree, self).add_edge(u, self.adversary)
		# 	super(RegularTree, self).add_edge(v, self.adversary)

		# def add_node(self, u, attr_dict = None, **attr):
		# 	super(RegularTree, self).add_node(u, attr)
		# 	if not (u == self.adversary):
		# 		super(RegularTree, self).add_edge(u, self.adversary)
		# 	self.max_node = max(self.nodes())

		# def generate_timestamp_dict(self):
		# 	''' Creates a dict with nodes as keys and timestamps as values '''
		# 	timestamp_dict = {}
		# 	for key in self.adversary_timestamps.keys():
		# 		print 'key', key
		# 		for node in self.adversary_timestamps[key]:
		# 			timestamp_dict[node] = key
		# 	return timestamp_dict


	def spread_message(self, first_spy_only = False):

		count = 0
		
		adversaries = [-(i+1) for i in range(self.theta)]

		# Empty the observed timestamps
		self.adversary_timestamps = {} 		# dictionary of adversary report time indexed by node
		self.received_timestamps = {}		# dictionary of message receipt time indexed by node

		# Initialize the process
		self.received_timestamps[self.source] = 0
		self.active = [self.source]
		self.infected = [self.source]

		stopping_time = self.spreading_time
		

		while self.active and count < stopping_time:
			count += 1
			new_boundary = []
			# cycle through the active nodes, and spread with an exponential clock
			for node in self.active:

				if self.degree(node) < (self.tree_degree):
					num_missing = (self.tree_degree) - self.degree(node)
					new_nodes = range(self.max_node + 1, self.max_node + num_missing + 1)
					self.add_edges(node, new_nodes)

				uninfected_neighbors = [neighbor for neighbor in self.neighbors(node) if neighbor not in self.infected]
				uninfected_neighbors += adversaries
				# print 'uninfected_neighbors', uninfected_neighbors, adversaries

				# random permutation of neighbors
				ordering = list(np.random.permutation(uninfected_neighbors))
				# print 'ordering', ordering
				signs = [item >= 0 for item in ordering]
				# print 'signs', signs

				# find the reporting time for node
				self.adversary_timestamps[node] = signs.index(False) + 1 + self.received_timestamps[node]
				# print 'timetsampts', self.adversary_timestamps

				if first_spy_only and (node == self.source):
					stopping_time = min(stopping_time, self.adversary_timestamps[node])
					# print 'stopping_time', stopping_time

				# assign the received timestamps for the other nodes
				for idx in range(len(ordering)):
					neighbor = ordering[idx]
					# if the node at time slot t is not a spy
					if neighbor >= 0:
						rx_timestamp = self.received_timestamps[node] + 1 + idx
						self.received_timestamps[neighbor] = rx_timestamp
						self.infected.append(neighbor)
						if rx_timestamp < stopping_time:
							self.active.append(neighbor)
				self.active.remove(node)


		# # ========================
		# t = 1
		# candidates = []
		
		
		# while (t <= self.spreading_time):
		# 	current_active = [item for item in self.active]
		# 	for node in current_active:
		# 		# Check that all the nodes have enough neighbors
		# 		if ((self.degree(node) < (self.tree_degree + 1) and self.has_edge(node, self.adversary)) or
		# 		   (self.degree(node) < (self.tree_degree) and (not self.has_edge(node, self.adversary)))): 
		# 			if (self.degree(node) < (self.tree_degree + 1) and self.has_edge(node, self.adversary)):
		# 				num_missing = (self.tree_degree + 1) - self.degree(node)
		# 			else:
		# 				num_missing = self.tree_degree - self.degree(node)
		# 			new_nodes = range(self.max_node + 1, self.max_node + num_missing + 1)
		# 			self.add_edges(node, new_nodes)
				
		# 		# print 'adjacency: ', self.edges()

		# 		# Spread to the active nodes' uninfected neighbors
		# 		uninfected_neighbors = self.get_uninfected_neighbors(node)
		# 		# print 'uninfected_neighbors', uninfected_neighbors, 'current_active', current_active, 'node', node
		# 		to_infect = random.choice(uninfected_neighbors)
		# 		self.infect_node(node, to_infect)
		# 		# print 'node ', node, ' infected ', to_infect
				
				
		# 		if to_infect == self.adversary:
		# 			# self.adversary_timestamps.append([node,t])
		# 			if (t in self.adversary_timestamps):
		# 				self.adversary_timestamps[t] += [node]
		# 			else:
		# 				self.adversary_timestamps[t] = [node]
		# 			# self.adversary_timestamps += [t]
		# 			# candidates += [node]
		# 	t += 1

		# # print 'timestamps:', self.adversary_timestamps

	

