# dataset_main.py
# Simulates diffusion and trickle on a dataset

from dataset_graph_rep import *
from dataset_estimators import *
from utils import *
import time
import numpy as np


lambda2s = xrange(1,20,2)
# lambda2s = [1.01]
check_ml = True

# filename = 'data/bitcoin.gexf'
# filename = 'data/random_regular.gexf'
# filename = 'data/tree_5.gexf'
# filename = 'data/8reg.gexf'
if filename == 'data/tree_5.gexf': 
	five_tree = 5
elif filename == 'data/8reg.gexf':
	five_tree = 8
else:
	five_tree = None

args = parse_arguments()

first_spy_only = (len(args.estimators) != 1) != (0 in args.estimators) # checks if the user only wants first-spy


accuracies_first = []
accuracies_local = []
accuracies_local_treelike = []
local_dist = []
accuracies_ml = []
# accuracies_first_diff = []
# count_first = 0


if args.spreading == DIFFUSION:
	if five_tree:
		spreading_time = 3
	else:
		spreading_time = 6
	G = DataGraphDiffusion(filename, spreading_time = spreading_time)
elif args.spreading == TRICKLE:
	spreading_time = 8
	G = DataGraphTrickle(filename, spreading_time = spreading_time)
	
	

# Dataset stats (comment out for actual runs)
degrees = [G.degree(n) for n in G.nodes()]
median_degree = np.median(degrees)
print 'Median degree is', median_degree
# print 'mean degree is ', np.mean(degrees)
# print 'median degree is ', np.median(degrees)
# print 'histogram is ', np.histogram(degrees, bins = range(100))
# exit(0)

for lambda2 in lambda2s:
	print 'On theta ', lambda2

	count_first = 0
	count_local = 0
	count_local_treelike = 0
	count_local_dist = 0
	count_ml = 0
	treelike_trials = 0
	for trial in range(args.trials):

		if (trial % 10) == 0:
			print 'On trial ', trial+1, ' out of ', args.trials

		if five_tree:
			source = random.choice([n for n in G.nodes() if G.degree(n)>=five_tree])
		else:
			source = random.choice(G.nodes())
	

		# Spread the message
		G.spread_message(source, first_spy_only = first_spy_only, num_corrupt_cnx = lambda2)

		# Estimate the source
		if FIRST_SPY in args.estimators:	
			est = FirstSpyEstimator(G, source, args.verbose, five_tree)
			result = est.estimate_source()
			acc = est.compute_accuracy(result)
			count_first += acc
			acc_fs = acc

		if LOCAL_OPT in args.estimators:
			# est = LocalEstimator(G, source, args.spreading, args.verbose, five_tree = five_tree)
			# est = TreeEstimator(G, source, args.spreading, args.verbose, five_tree = five_tree)
			est = SecondReportEstimator(G, source, args.spreading, args.verbose)
			result, treelike, dist = est.estimate_source(spreading_time)
			acc = est.compute_accuracy(result)
			count_local += acc
			if treelike:
				count_local_treelike += acc
				treelike_trials += 1
				count_local_dist += dist

			if acc != acc_fs:
				print 'Discrepancy!'
				print 'FS Result: ', acc_fs
				print 'Local result:', acc
				# break

		# if LOCAL_OPT in args.estimators:
		# 	est = MLEstimator(G, spreading = args.spreading, args.verbose)
		# 	result = est.estimate_source()
		# 	acc = est.compute_accuracy(G.source, result)
		# 	count_local += acc

	# accuracies_first += [float(count_first) / args.trials]
	accuracies_first += [float(count_first) / args.trials]
	accuracies_local += [float(count_local) / args.trials]
	accuracies_local_treelike += [float(count_local_treelike) / max(treelike_trials,1)]
	local_dist += [float(count_local_dist) / max(treelike_trials,1)]
	accuracies_ml += [float(count_ml) / args.trials]

print 'The first-spy estimator accuracy: ', accuracies_first
print 'The local estimator accuracy: ', accuracies_local
print 'The local estimator accuracy (treelike instances): ', accuracies_local_treelike
print 'The local estimator hop distance (treelike instances): ', local_dist