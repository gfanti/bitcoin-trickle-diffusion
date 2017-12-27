# dataset_estimators.py
import networkx as nx
import random
from sortedcontainers import SortedDict
import itertools
import math
import numpy as np
from utils import *

class Estimator(object):

	def  __init__(self, G, source, verbose = False):
		self.G = G
		self.verbose = verbose
		self.source = source

	def compute_accuracy(self, candidates):
		if self.source in candidates:
			return 1.0 / len(candidates)
		else:
			return 0.0	

class FirstSpyEstimator(Estimator):

	def __init__(self, G, source, verbose = False, regular_degree = None):
		super(FirstSpyEstimator, self).__init__(G, source, verbose)
		self.regular_degree = regular_degree

	def estimate_source(self):
		if self.verbose:
			print 'Edges: ', self.G.edges()
			# print 'Timestamps: ', self.G.adversary_timestamps
			print 'Receive Timestamps: ', self.G.received_timestamps

		if self.regular_degree:
			initial = [node for node in self.G.nodes() if self.G.degree(node) >= self.regular_degree and node in self.G.adversary_timestamps]
			min_ts_initial = min([self.G.adversary_timestamps[t] for t in initial])
			candidates = [node for node in initial
								if self.G.adversary_timestamps[node] == min_ts_initial]
		else:
			candidates = [node for node in self.G.adversary_timestamps.keys() 
								if self.G.adversary_timestamps[node] == min(self.G.adversary_timestamps.values())]
		
		# if self.verbose:
		print 'First spy candidates: ', [(candidate, self.G.adversary_timestamps[candidate]) for candidate in candidates]

		# # For lower bound in Sigemtrics submission, uncomment the next two lines
		# if len(candidates) > 1:
		# 	candidates = []

		return candidates

class GraphBasedEstimator(Estimator):
	def __init__(self, G, source, spreading, verbose = False, regular_degree = None):
		super(GraphBasedEstimator, self).__init__(G, source, verbose)
		self.spreading = spreading
		self.theta = self.G.num_corrupt_cnx
		self.regular_degree = regular_degree

	def estimate_source(self, est_time, median_degree = 1):
		if not self.regular_degree:
			initial = [node for node in self.G.adversary_timestamps.keys()]
		else:
			initial = [node for node in self.G.adversary_timestamps.keys() if self.G.degree(node) >= self.regular_degree]
		# initial = [node for node in self.G.nodes()]
		# print 'initial', initial
		if self.spreading == DIFFUSION:
			likelihoods = {}
			for candidate in initial:
				likelihoods[candidate] = self.compute_likelihood_diffusion(candidate, est_time)

			candidates = [candidate for candidate in likelihoods.keys() if likelihoods[candidate] == max(likelihoods.values())]
			print 'candidates', [(c,likelihoods[c], self.G.degree(c)) for c in candidates]
			print 'source likelihood', (self.source, likelihoods[self.source], self.G.adversary_timestamps[self.source], self.G.degree(self.source))
			source2neighbors = [self.G.infected_by_source[n] for n in self.G.neighbors(self.source) if n in self.G.adversary_timestamps]
			if all(source2neighbors):
				print 'neighbors infected by source?', [(n, self.G.infected_by_source[n], self.G.adversary_timestamps[n]) \
									for n in self.G.neighbors(self.source) if n in self.G.adversary_timestamps]

		elif self.spreading == TRICKLE:
			pass

		return (candidates, all(source2neighbors), nx.shortest_path_length(self.G, candidates[0], self.source))


class TreeEstimator(GraphBasedEstimator):
	def __init__(self, G, source, spreading, verbose = False, regular_degree = False):
		super(TreeEstimator, self).__init__(G, source, spreading, verbose, regular_degree)
		
	
	def compute_likelihood_diffusion(self, node, est_time):
		''' Computes a local likelihood estimate based on diffusion spreading. 
		Finds the most likely path, and then computes the likelihood based on that.'''

		likelihood = self.compute_likely_tree(node, est_time)

		return likelihood

	def compute_likely_tree(self, node, est_time):

		source_time = self.G.adversary_timestamps[node]
		infected = [node]
		# if node == self.source:
		# 	print 'source time', node, ': ', source_time

		# Compute the likelihood of the candidate source
		likelihood = np.log(self.theta * np.exp(-self.theta * source_time))

		# Trace the likelihood of the ML path
		# if node == self.source:
		# 	print 'self n neighbors', node, ': ', self.G.neighbors(node)
		boundary = [(node, source_time, n, self.G.adversary_timestamps[n]) for n in self.G.neighbors(node) if n in self.G.adversary_timestamps]
		count = 0
		num_nodes = nx.number_of_nodes(self.G)
		while boundary and (count <= num_nodes):
			count += 1
			# Sort by timestamp
			boundary.sort(key = lambda x : x[3] - x[1])
			# if node == self.source:
			# 	print 'boundary', boundary
			item = boundary.pop(0)	# remove the first element
			src, src_timestamp, target, target_timestamp = item
			# Compute the likelihood of the fired edge
			likelihood += np.log(self.prob_adjacent_timestamps(src_timestamp, target_timestamp))
			# Compute the likelihoods of the other edges to target firing later than t2
			for edge in boundary:
				if edge[2] == target:
					likelihood += np.log(self.prob_adjacent_timestamps_ccdf(edge[1], edge[3]))

			# if node == self.source:
			# 	print 'infecting ',target, ': ', target_timestamp
			infected.append(target)
			# Add the remaining neighbors who have reported
			boundary += [(target, target_timestamp, n, self.G.adversary_timestamps[n]) \
							for n in self.G.neighbors(target) if n in self.G.adversary_timestamps and n not in infected]
			# Remove the infected edges from the boundary
			boundary = [b for b in boundary if b[0] in infected and b[3] not in infected]
		return likelihood

	def prob_adjacent_timestamps(self, t1, t2):
		'''Computes the likelihood of seeing t1 next to t2, given that node 1 infected 2'''
		x = t2 - t1
		if x < 0:
			return self.theta / 2 / (self.theta - 1) * np.exp(self.theta * x)
		else:
			return pow(self.theta,2) / (pow(self.theta,2)-1) * np.exp(-x) - self.theta \
							/ (self.theta - 1) / 2 * np.exp(-self.theta * x)

	def prob_adjacent_timestamps_ccdf(self, t1, t2):
		''' Returns the likelihood that the edge from 1 to 2 fired LATER than t2-t1'''
		x = t2 - t1
		if x < 0:
			return 1.0 - 0.5 * np.exp(self.theta * x) / (self.theta + 1)
		else:
			return 1.0 - 0.5 * ((1 + self.theta * (1 - np.exp(-x)))/(self.theta + 1) + (self.theta * \
							(1-np.exp(-x)) - (1 - np.exp(-self.theta * x))) / (self.theta - 1) )

class SecondReportEstimator(GraphBasedEstimator):
	def __init__(self, G, source, spreading, verbose = False, regular_degree = False):
		super(SecondReportEstimator, self).__init__(G, source, spreading, verbose, regular_degree)
		
	
	def compute_likelihood_diffusion(self, node, est_time):
		''' Computes the likelihood of the smallest timestamp among node's neighbors. 
		Finds the most likely path, and then computes the likelihood based on that.'''

		source_time = self.G.adversary_timestamps[node]
		infected = [node]

		# Compute the likelihood of the candidate source
		likelihood = np.log(self.theta) - self.theta * source_time

		neighbors = [(n, self.G.adversary_timestamps[n]) for n in self.G.neighbors(node) if n in self.G.adversary_timestamps]
		# print 'neighbors', neighbors
		neighbors.sort(key = lambda x: x[1])
		# print 'num nodes', self.G.number_of_nodes()
		# print '\n likelihood for node ', (node, source_time)
		# print 'neighbors', neighbors
		if neighbors:
			neighbor, neighbor_timestamp = neighbors.pop(0)
			likelihood += np.log(self.first_report_likelihood(node, neighbor_timestamp))
			# print '(neighbor, likelihood)', (neighbor, likelihood)
		else:
			likelihood += np.log(self.first_report_likelihood(node,est_time))

		return likelihood

	def first_report_likelihood(self, node, t):
		''' Computes the likelihood of the smallest timestamp next to node being t'''
		d = self.G.degree(node)
		lik = self.theta * d / pow(self.theta-1, d) * pow(self.theta * np.exp(-t) - np.exp(-self.theta * t), d-1) * \
				(np.exp(-t) - np.exp(-self.theta * t))
		return lik

class LocalEstimator(GraphBasedEstimator):
	def __init__(self, G, source, spreading, verbose = False, regular_degree = None):
		super(LocalEstimator, self).__init__(G, source, spreading, verbose, regular_degree)
		
	def compute_likelihood_diffusion(self, node, est_time, median_degree = 1):
		''' Computes a local likelihood estimate based on diffusion spreading'''
		source_time = self.G.adversary_timestamps[node]
		neighbors = self.G.neighbors(node)
		infected_neighbors = [n for n in neighbors if n in self.G.adversary_timestamps.keys()]
		uninfected_neighbors = [n for n in neighbors if n not in infected_neighbors]

		likelihood = 0

		# Add the terms that depend only on uninfected neighbors
		likelihood += len(uninfected_neighbors) * (-est_time + np.log(1 + 1/(self.theta-1)*(1-np.exp(-est_time * (self.theta - 1)))))
		
		
	  	# Add the terms that depend on infected neighbor timestamps
	  	likelihood += len(infected_neighbors) * np.log(self.theta / (self.theta - 1))
	  	for neighbor in infected_neighbors:
	  		likelihood += np.log(np.exp(-self.G.adversary_timestamps[neighbor]) - np.exp(-self.G.adversary_timestamps[neighbor]*self.theta))

  		# print '\n likelihood for node ', (node, source_time)
  		ns = [(neighbor, self.G.adversary_timestamps[neighbor]) for neighbor in infected_neighbors]
  		ns.sort(key = lambda x: x[1])
		# print 'neighbors', ns

	  	# print 'neighbor terms', likelihood
		
  		
  		# # Normalize by median degree
  		# likelihood = likelihood / float(self.G.degree(node))

  		# Add the terms that depend only on the candidate source
		if node in self.G.adversary_timestamps.keys():	# if the candidate is infected
			likelihood += np.log(self.theta) - self.theta * source_time #* 400
		else:
			likelihood -= self.theta * est_time
		# print 'final', likelihood


		return likelihood



