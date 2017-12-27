# estimators.py
# Estimators that are (mainly) designed for trees

import networkx as nx
import random
from sortedcontainers import SortedDict
import itertools
import math
import numpy as np



class Estimator(object):

	def  __init__(self, G, verbose = False):
		self.G = G
		self.verbose = verbose

	def compute_accuracy(self, source, candidates):
		if source in candidates:
			return 1.0 / len(candidates)
		else:
			return 0.0	


class DiffusionEstimator(Estimator):
	def __init__(self, G, verbose = False):
		super(DiffusionEstimator, self).__init__(G, verbose)

class FirstSpyEstimator(Estimator):

	def __init__(self, G, verbose = False):
		super(FirstSpyEstimator, self).__init__(G, verbose)

	def estimate_source(self):


		if self.verbose:
			print 'Edges: ', self.G.edges()
			# print 'Timestamps: ', self.G.adversary_timestamps
			print 'Receive Timestamps: ', self.G.received_timestamps


		candidates = [node for node in self.G.adversary_timestamps.keys() 
							if self.G.adversary_timestamps[node] == min(self.G.adversary_timestamps.values())]
		
		if self.verbose:
			print 'First spy candidates: ', candidates

		# For lower bound, uncomment the next two lines
		if len(candidates) > 1:
			candidates = []

		return candidates

# class FirstSpyDiffusionEstimator(DiffusionEstimator):

# 	def __init__(self, G, verbose = False):
# 		super(FirstSpyDiffusionEstimator, self).__init__(G, verbose)

# 	def estimate_source(self):

# 		if self.verbose:
# 			print 'Edges: ', self.G.edges()
# 			# print 'Timestamps: ', self.G.adversary_timestamps
# 			print 'Receive Timestamps: ', self.G.received_timestamps


# 		candidates = [node for node in self.G.adversary_timestamps.keys() 
# 							if self.G.adversary_timestamps[node] == min(self.G.adversary_timestamps.values())]
	
# 		if self.verbose:
# 			print 'First spy candidates: ', candidates

# 		return candidates

class PrunedDiffusionEstimator(DiffusionEstimator):
	'''Some nodes have the message but haven't reported yet. This estimator prunes off the boundary 
	nodes that havent reported yet.'''

	def __init__(self, G, verbose = False):
		super(PrunedDiffusionEstimator, self).__init__(G, verbose)
		self.T = G.copy()
		self.prune_boundary()

	def prune_boundary(self):
		''' Prunes the boundaries of the tree until there are no more unreported nodes 
		at the boundary'''

		count = 0;

		# print 'Edges', self.T.edges()
		# print 'Reporting times', self.T.adversary_timestamps

		while True:

			leaves = [x for x in self.T.nodes_iter() if self.T.degree(x) == 1]
			flag = False
			for leaf in leaves:
				# print 'Checking leaf ', leaf 
				# print '    had rx time ', self.T.received_timestamps[leaf]
				if (leaf not in self.T.adversary_timestamps) or (self.T.adversary_timestamps[leaf] > self.T.spreading_time):
					self.T.remove_node(leaf)
					flag = True
					count += 1
			if not flag:
				break
		# print 'Pruned ', count, ' nodes.'


class RumorCentralityEstimator(PrunedDiffusionEstimator):
	'''We expect the timestamps to increase in a cone around the true source. This estimator
	tries to quantify the extent to which that happens--it penalizes nodes by the square of 
	how much their adversarial report times differ from the expected timestamp'''

	def __init__(self, G, verbose = False):
		super(RumorCentralityEstimator, self).__init__(G, verbose)


	def estimate_source(self):
		# initialize the messages vector to likelihood 1

		candidates = self.rumor_centrality()

		return candidates

	def rumor_centrality(self):
		# computes the estimate of the source based on rumor centrality
		# choose an arbitrary initial index that has the message
		initial_node = self.T.nodes()[0]       

		# Initialize messages to ones
		up_messages = {}
		down_messages = {}
		for i in self.T.nodes():
			up_messages[i] = 1
			down_messages[i] = 1

		up_messages = self.rumor_centrality_up(up_messages, initial_node, initial_node)
		down_messages = self.rumor_centrality_down(up_messages, down_messages, initial_node, initial_node)

		# Get the indices of the maximum values of the down messages
		max_down = max(down_messages.values())
		max_down_ind = [key for key,value in down_messages.items() if value == max_down]
		return max_down_ind

	def rumor_centrality_up(self, up_messages, calling_node, called_node):
		if called_node == calling_node:
			for i in self.T.neighbors(called_node):
				up_messages = self.rumor_centrality_up(up_messages, called_node, i)
		elif self.T.degree(called_node) == 1:   # leaf node
			up_messages[calling_node] += 1 # check
		else:
			for i in self.T.neighbors(called_node):
				if i != calling_node:
					up_messages = self.rumor_centrality_up(up_messages, called_node, i)
			up_messages[calling_node] += up_messages[called_node]
		return up_messages        

	def rumor_centrality_down(self, up_messages, down_messages, calling_node, called_node):
		if called_node == calling_node:
			for i in self.T.neighbors(called_node):
				down_messages = self.rumor_centrality_down(up_messages, down_messages, called_node, i)   
		else:
			# print("calling_node", calling_node, "length", len(down_messages))
			# print("down calling", calling_node, down_messages[calling_node])
			down_messages[called_node] = down_messages[calling_node]*(float(up_messages[called_node])/(self.T.number_of_nodes()-up_messages[called_node]))
			for i in self.T.neighbors(called_node):
				if i != calling_node:
					down_messages = self.rumor_centrality_down(up_messages, down_messages, called_node, i)
		return down_messages 

class ReportingCentralityEstimator(PrunedDiffusionEstimator):
	'''We expect the timestamps to increase in a cone around the true source. This estimator
	tries to quantify the extent to which that happens--it penalizes nodes by the square of 
	how much their adversarial report times differ from the expected timestamp'''

	def __init__(self, G, verbose = False):
		super(ReportingCentralityEstimator, self).__init__(G, verbose)


	def estimate_source(self):
		# initialize the messages vector to likelihood 1

		candidates = self.reporting_centrality()

		return candidates

	def reporting_centrality(self):
		# computes the estimate of the source based on rumor centrality
		# choose an arbitrary initial index that has the message

		reporting_centers = []

		for node in self.T.nodes():
			counts = []
			for neighbor in self.T.neighbors(node):
				counts += [self.count_reporting_nodes(node, neighbor)]
			Y = sum(counts)
			if all([count < Y/2.0 for count in counts]):
				reporting_centers += [node]
		
		return reporting_centers

	        

	def count_reporting_nodes(self, source, target):
		
		total = 0

		if target in self.T.adversary_timestamps:
			total += 1

		if self.T.degree(target) == 1:
			return total

		children = [neighbor for neighbor in self.T.neighbors(target) if not neighbor == source]
		for child in children: 
			total += self.count_reporting_nodes(target, child)
		
		return total

class ConeDiffusionEstimator(PrunedDiffusionEstimator):
	'''We expect the timestamps to increase in a cone around the true source. This estimator
	tries to quantify the extent to which that happens--it penalizes nodes by the square of 
	how much their adversarial report times differ from the expected timestamp'''

	def __init__(self, G, verbose = False):
		super(ConeDiffusionEstimator, self).__init__(G, verbose)


	def estimate_source(self):

		if self.verbose:
			print 'adversary timestamps:', self.T.adversary_timestamps

		if not self.T.nodes():
			return []

		score = [0 for i in range(len(self.T.nodes()))]
		score = {}
		for node in self.T.nodes():

			if node in self.T.adversary_timestamps:
				t_ref = self.T.adversary_timestamps[node]	# reference time at which node reported to source
			else:
				t_ref = self.T.spreading_time

			score[node] = self.compute_score(node, t_ref)

		scores = np.array(score.values())
		nodes = score.keys()

		if self.verbose:
			print 'scores:', scores
			print 'nodes:', nodes
		indices = np.where(scores == scores.min())
		
		candidates = [nodes[i] for i in indices[0]]
	
		if self.verbose:
			print 'cone candidates:', candidates
		
		return candidates

	def compute_score(self, node, t_ref):
		
		score = 0
		path_lengths = nx.shortest_path_length(self.T, node)

		for v in self.T.nodes():
			if v == node:
				continue

			hop_dist = path_lengths[v]
			expected_time = t_ref + self.T.lambda1 * hop_dist
			sd = np.power(hop_dist + (2 / np.square(self.T.lambda2)), 1)	# standard deviation of erlang RV, sum of hop_dist exponentials
			if v in self.T.adversary_timestamps:
				reported_time = self.T.adversary_timestamps[v]
			else:
				reported_time = self.T.spreading_time 
			score += np.square((reported_time - expected_time) / sd)

		return score

class NeighborConeDiffusionEstimator(ConeDiffusionEstimator):
	'''We expect the timestamps to increase in a cone around the true source. This estimator
	tries to quantify the extent to which that happens--it penalizes nodes by the square of 
	how much their adversarial report times differ from the expected timestamp'''

	def __init__(self, G, verbose = False):
		super(NeighborConeDiffusionEstimator, self).__init__(G, verbose)

	def compute_score(self, node, t_ref):
		score = 0
		path_lengths = nx.shortest_path_length(self.T, node)

		''' (1) Get spanning tree
			(2) Do a depth-first traversal of spanning tree
			(3) Compute the distance from the cone, but re-calibrate the cone at each observation
		'''
		dfs_tree = nx.dfs_tree(self.T, node)

		for n in dfs_tree.nodes():
			if (not n in self.T.adversary_timestamps) or (n == node):
				continue

			hop_dist = 1	# number of hops since last adversary observation
			cur_node = dfs_tree.predecessors(n)[0]
			while True:
				if (cur_node in self.T.adversary_timestamps) or (cur_node == node):
					break
				# Move up the tree until you get a timestamp or hit the root
				cur_node = dfs_tree.predecessors(cur_node)[0]
				hop_dist += 1

			expected_time = t_ref + self.T.lambda1 * hop_dist
			sd = np.power(hop_dist + (2 / np.square(self.T.lambda2)), 0.5)	# standard deviation of erlang RV, sum of hop_dist exponentials
				
			if cur_node in self.T.adversary_timestamps:
				reported_time = self.T.adversary_timestamps[cur_node]
			else:
				reported_time = self.T.spreading_time 
			score += np.square((reported_time - expected_time) / sd)

		return score

class LocalMLDiffusionEstimator(ConeDiffusionEstimator):
	'''Compute the local ML rule, but only in the neighborhood of each node'''

	def __init__(self, G, locality = 2, verbose = False):
		super(LocalMLDiffusionEstimator, self).__init__(G, verbose)
		self.locality = locality

	def compute_score(self, node, t_ref):
		score = 0
		path_lengths = nx.shortest_path_length(self.T, node)

		''' (1) Get spanning tree
			(2) Compute the likelihood of a local cone
		'''
		dfs_tree = nx.dfs_tree(self.T, node)

		for n in dfs_tree.nodes():
			if (not n in self.T.adversary_timestamps) or (n == node) or (path_lengths[n] > locality):
				continue

			hop_dist = 1	# number of hops since last adversary observation
			cur_node = dfs_tree.predecessors(n)[0]
			while True:
				if (cur_node in self.T.adversary_timestamps) or (cur_node == node):
					break
				# Move up the tree until you get a timestamp or hit the root
				cur_node = dfs_tree.predecessors(cur_node)[0]
				hop_dist += 1

			expected_time = t_ref + self.T.lambda1 * hop_dist
			sd = np.power(hop_dist + (2 / np.square(self.T.lambda2)), 0.5)	# standard deviation of erlang RV, sum of hop_dist exponentials
				
			if cur_node in self.T.adversary_timestamps:
				reported_time = self.T.adversary_timestamps[cur_node]
			else:
				reported_time = self.T.spreading_time 
			score += np.square((reported_time - expected_time) / sd)

		return score


class GossipEstimator(Estimator):

	def __init__(self, G, verbose = False):
		super(GossipEstimator, self).__init__(G, verbose)

	def get_starting_set(self, timestamp_dict):
		# Gets the set of nodes within an appropriate radius of the 
		# nodes that get the message first

		min_timestamp, candidates_first_spy = self.G.adversary_timestamps.items()[0]
		candidates = set(candidates_first_spy)

		# Then look in an appropriate radius of the first timestamp...
		cand_neighborhood = []
		for candidate in candidates:
			neighborhood = set([candidate])
			for i in range(min_timestamp - 1):
				# print 'neighboring set',set(self.G.get_neighbors(candidates))
				neighborhood = neighborhood.union(set(self.G.get_neighbors(neighborhood)))
			cand_neighborhood += [neighborhood]
		candidates = set.intersection(*cand_neighborhood)
		if self.G.adversary in candidates:
			candidates.remove(self.G.adversary)
		return candidates


class MLEstimator(GossipEstimator):

	def __init__(self, G, verbose = False):
		super(MLEstimator, self).__init__(G, verbose)

	def estimate_source(self):
		''' Returns the list of nodes that could feasibly be
		the true source (THIS IS NOT THE ML ESTIMATOR)'''

		# Make sure there are timestamps
		if not any(self.G.adversary_timestamps):
			print 'No timestamps found.'
			return []

		# print 'timestamps are ', self.G.adversary_timestamps
		# print 'adjacency is ', self.G.edges()
		


		# Find the list of eligible nodes, cut 1
		# Start with the first-spy estimate...
		timestamp_dict = self.G.generate_timestamp_dict()
		candidates = self.get_starting_set(timestamp_dict)

		# print 'candidates are', candidates, 'before pruning'
		# print 'timestamp_dict', timestamp_dict
		# self.G.draw_plot()
		# Now check if each of these nodes in the radius is eligible

		final_candidates = [i for i in candidates]
		for candidate in candidates:
			# print 'candidate', candidate
			valid = True
			# print 'candidate', candidate, ' has timestamp', timestamp_dict[candidate]
			for node in candidates:
				timestamp = timestamp_dict[node]
				# print 'node', node
				# print 'timestamp',timestamp, 'lb', self.min_timestamp(candidate, node), 'ub', self.max_timestamp(candidate, node)
				if not self.check_node(candidate, node, timestamp):
					valid = False
					# print 'FAIL candidate', candidate
					break
			if not valid:
				final_candidates.remove(candidate)

		# print 'ml candidates are', final_candidates
		return final_candidates


	def check_node(self, source, target, timestamp):
		''' Checks if target's timestamp is eligible'''
		lowerbound = (timestamp >= self.min_timestamp(source, target))
		upperbound = (timestamp <= self.max_timestamp(source, target))
		return (lowerbound and upperbound)

	def min_timestamp(self, source, target):
		return (nx.shortest_path_length(self.G, source = source, target = target) + 1)

	def max_timestamp(self, source, target):
		d = self.G.tree_degree
		pathlength = nx.shortest_path_length(self.G, source = source, target = target)

		if pathlength == 0:
			return (d + 1)
		return (d + 1) + (d * pathlength)


class MLEstimatorMP(GossipEstimator):

	def __init__(self, G, verbose = False):
		super(MLEstimatorMP, self).__init__(G, verbose)
		self.timestamp_dict = None
		self.rx_time = {}
		self.count_dict = {}
		self.adversary = self.G.adversary
		

	def estimate_source(self):
		''' Returns the list of nodes that have the maximumum likelihood of being
		the true source'''

		# Make sure there are timestamps
		if not any(self.G.adversary_timestamps):
			print 'No timestamps found.'
			return []


		# Get the starting set of nodes
		self.timestamp_dict = self.G.generate_timestamp_dict()
		# try not updating boundary nodes
		# ---------------------------------self.update_boundary_nodes()
		candidates = self.get_starting_set(self.timestamp_dict)

		if self.verbose:
			# print 'timestamps are ', self.G.adversary_timestamps
			print 'adjacency is ', self.G.edges()
			print 'candidates are', candidates
			print 'timestamps are: ', self.timestamp_dict
		# self.G.draw_plot()
		# Now check if each of these nodes in the radius is eligible

		# Now compute the number of paths of viable infection for each node
		final_candidates = [i for i in candidates]

		# Modify the graph to include "boundary nodes"

		counts = []

		# print 'adjacency', self.G.edges()


		# print 'candidates: ', candidates, 'timestamps', self.timestamp_dict, '\n'
		for candidate in candidates:
			if self.verbose:
				print '\nprocessing candidate ', candidate, '\n'

			self.feasible = True
			# -----Run the message-passing------
			count = self.pass_down_messages(candidate, candidate)
			
			if self.verbose:
				print 'candidate', candidate, ' has count ', count
			
			counts += [count]

		if self.verbose:
			print 'candidates counts are ', zip(candidates, counts)

		final_candidates = [candidate for (candidate, score) in zip(candidates, counts) if score == max(counts)]
		return final_candidates


	def update_boundary_nodes(self):
		''' Make it look like all the nodes at the boundary have timestamp T+1 '''
		for n in self.G.nodes():
			# if self.G.node[n]['infected'] and (n not in self.timestamp_dict):
			# 	self.timestamp_dict[n] = self.G.spreading_time + 1
			if (not n == self.G.adversary) and (n not in self.timestamp_dict):
				self.timestamp_dict[n] = self.G.spreading_time + 1

	def get_tree_neighbors(self, node, remove_item = None):
		''' Get a node's neighbors that are infected and not the adversary, except 
		    for item remove_item'''

		neighbors = [n for n in self.G.neighbors(node) if 
						(self.G.node[n]['infected'] == True) and
						not (n == self.adversary) and
						not (n == remove_item) and 
						(n in self.timestamp_dict)]
		return neighbors
			
	def compute_tx_time(self, target, source_flag = False):
		# Otherwise, create the list of possible rx times for the children of target
		tx_time = set()

		if self.verbose:
			print 'self.rx_time[', target, '] = ', self.rx_time[target]
		for t in self.rx_time[target]:
			if source_flag:
				tx_time.update([t+i for i in range(1, self.G.tree_degree + 2)])
			else:
				tx_time.update([t+i for i in range(1, self.G.tree_degree + 1)])
		tx_time = list(tx_time)
		try:
			tx_time.remove(self.timestamp_dict[target])
		except:
			pass
		return tx_time

	def pass_down_messages(self, source, target):
		''' Pass messages down the tree with the feasible set of timestamps'''

		# If source = target, then we're at the root
		if source == target:
			self.rx_time[source] = [0]
			if self.verbose:
				print 'self.rx_time[', source, '] = ', self.rx_time[source]

			# Get the child nodes
			child_nodes = self.get_tree_neighbors(target)

		else:
			# Make sure that there's an edge between source and target
			if not self.G.has_edge(source, target):
				return 0		

			# Identify the target's child nodes
			child_nodes = self.get_tree_neighbors(target, source)

		# Initialize the down messages		
		self.count_dict[target] = {}
		for rx_time in self.rx_time[target]:
			self.count_dict[target][rx_time] = 0
		# print 'dict for target ', target, 'looks like dis', self.count_dict[target]

		# If we're at a leaf, stop passing the message
		if not child_nodes:
			# Now set the up-messages to unit value
			for item in self.rx_time[target]:
				self.count_dict[target][item] = 1 # number of permutations possible (i.e. 1)
			# print 'at a leaf', target
			return

		# Otherwise, create the list of possible rx times for the children of target
		tx_time_baseline = self.compute_tx_time(target, source == target)
		# print 'at first, tx_time was ', tx_time_baseline

		tx_time_list = []
		# Pass the message to child nodes
		for child in child_nodes:
			# Remove any nonphysical timestamps in the rx_time, 
			# i.e. rx_times that happen after the  target's observed timestamp
			
			tx_time = [i for i in tx_time_baseline]

			# Prune the possible tx_times based on the observed timestamps at the receiver
			# print 'child ', child, 'has timestamp', self.timestamp_dict[child], 'and our spreading_time is ', self.G.spreading_time
			# if (self.timestamp_dict[child] > self.G.tree_degree):
			# 	tx_time = [i for i in tx_time if (i >= self.timestamp_dict[child] - self.G.tree_degree)]
			# else:
			tx_time = [i for i in tx_time if (i >= self.timestamp_dict[child] - self.G.tree_degree)
					   					 and (i < self.timestamp_dict[child])]
			# print 'pruned tx_time for child', child, ' is:',tx_time

			self.rx_time[child] = tx_time
			if self.verbose:
				print 'self.rx_time[', child, '] = ', tx_time
			
			# If there are no valid timestamps, then this candidate is not feasible
			if not tx_time:
				return

			# Add the node's feasible rx_times to the list
			tx_time_list += [tx_time]
			self.pass_down_messages(target, child)



		# Aggregate the messages from the children and pass it up the chain
		# order of tx_time_list is [child_nodes target]
		self.aggregate_messages(target, child_nodes, tx_time_list)
		# print 'after dict for target ', target, 'is ', self.count_dict[target]
			
		# print 'degree', self.G.tree_degree

		# if we're at the root, sum up all the elements in the dictionary
		if (source == target):
			# print 'counts from neighbors', self.count_dict[source].keys(), self.count_dict[source].values()
			return sum(self.count_dict[source].values())
		

	def aggregate_messages(self, node, neighbors, tx_time_list):
		# Aggregate the messages from the children and pass it up the chain
		# order of tx_time_list is [child_nodes target]
		# print 'target', node, 'child_nodes', neighbors, 'tx_time_list', tx_time_list
		coordinates = neighbors + [node]	# [children parent]
		tx_time_list += [self.rx_time[node]]  # [[child 1's rx times],...,[child d-1's rx times], [self rx_times]]
		# print 'tx_time_list', tx_time_list
		tuple_list = [list(item) for item in list(itertools.product(*tx_time_list)) if len(set(item)) == len(item)]	# all items in the tuple distinct
		tuple_list = [item for item in tuple_list if ((max(item) - min(item)) <= (self.G.tree_degree+1)) ]	# all timestamps in a tuple are at most d apart
		
		# print 'tuple_list', tuple_list


		# Count the number of paths that use each candidate arrival time at the source
		for item in tuple_list:
			# Multiply the counts associated with each coordinate
			# computes list for each tuple index of the number of permutations in which neighbor i receives the message at time item[i]
			m = [self.count_dict[neighbors[i]][item[i]] for i in range(len(item)-1)] 
			self.count_dict[node][item[-1]] += np.prod(m)	# computes the product of those permutations
			# print 'm is ', m

			# Sum the logs of the counts
			# m = [math.log(self.count_dict[neighbors[i]][item[i]]) for i in range(len(item)-1)]
			# # Instead of multiplying the values, we add the sum
			# self.count_dict[node][item[-1]] += sum(m)

		if self.verbose:
			print 'up-counts for ', node, 'is ', self.count_dict[node]
			





