# main.py
# Simulates diffusion and trickle on a regular tree
# Can check ML estimator and/or first-spy estimators

from graph_rep import *
from estimators import *
from utils import *
import time

if __name__ == "__main__":
	
	# thetas = [1]
	# thetas = xrange(1,20,2)
	theta = 1  # number of connections to the eavesdropper per node
	# degrees = xrange(7,8,1)
	degrees = [2,3,4,5,6] # range of regular tree degrees to test
	
	check_ml = False # use the ML estimator? this can be very slow


	args = parse_arguments()



	trickle = False
	diffusion = not trickle


	accuracies_first = []
	accuracies_first_diff = []
	accuracies_ml = []
	accuracies_rc = []
	accuracies_cone = []
	accuracies_neighbor_cone = []

	if args.measure_time:
		start = time.time()
		end = start

	# spreading_times = [2,225,1.5]

	# cnt = 0
	for degree in degrees:
	# for theta in thetas:

		# spreading_time = spreading_times[cnt]
		# cnt += 1

		# Set the spreading time
		if trickle:
			# We limit the spreading time to degree + 1 for efficiency
			# 	if the first spy estimator succeeds, it will always do 
			#	so by this time
			spreading_time = degree + 1
		else:
			spreading_time = 2

		print 'On degree ', degree
		print 'On theta', theta

		count_first = 0
		count_first_diff = 0
		count_ml = 0
		count_rc = 0
		count_cone = 0
		count_neighbor_cone = 0

		for i in range(args.trials):
			if (i % 100) == 0:
				print 'On trial ', i+1, ' out of ', args.trials

			if trickle:
				# Trickle trials
				G = RegularTreeTrickle(degree,spreading_time, num_corrupt_cnx = theta)
				G.spread_message(first_spy_only = (not check_ml))
				
				# First spy estimator
				est_first = FirstSpyEstimator(G)
				result_first = est_first.estimate_source()
				acc_first = est_first.compute_accuracy(G.source, result_first)
				count_first += acc_first

				if check_ml:
					# ML estimator general
					est_ml = MLEstimatorMP(G, args.verbose)
					result_ml = est_ml.estimate_source()
					acc_ml = est_ml.compute_accuracy(G.source, result_ml)
					count_ml += acc_ml

			if diffusion:
				# Diffusion trials
				G = RegularTreeDiffusion(degree, spreading_time, theta)
				G.spread_message(first_spy_only = (not check_ml))

				# G.draw_plot()

				# First spy estimator
				est_first = FirstSpyEstimator(G, args.verbose)
				result_first = est_first.estimate_source()
				acc_first = est_first.compute_accuracy(G.source, result_first)
				count_first_diff += acc_first

				# # Rumor centrality estimator
				# est_rc = RumorCentralityEstimator(G, args.verbose)
				# result_rc = est_rc.estimate_source()
				# acc_rc = est_rc.compute_accuracy(G.source, result_rc)
				# count_rc += acc_rc

				if check_ml:
					# Reporting centrality estimator [OVERWRITES RUMOR CENTRALITY]
					# 	This is not actually ML, but it is the estimator analyzed 
					# 	in our paper
					est_rc = ReportingCentralityEstimator(G, args.verbose)
					result_rc = est_rc.estimate_source()
					acc_rc = est_rc.compute_accuracy(G.source, result_rc)
					count_rc += acc_rc

				# # Cone estimator
				# est_cone = ConeDiffusionEstimator(G, args.verbose)
				# result_cone = est_cone.estimate_source()
				# acc_cone = est_cone.compute_accuracy(G.source, result_cone)
				# count_cone += acc_cone

				# # Neighbor cone estimator
				# est_cone = NeighborConeDiffusionEstimator(G, args.verbose)
				# result_cone = est_cone.estimate_source()
				# acc_cone = est_cone.compute_accuracy(G.source, result_cone)
				# count_neighbor_cone += acc_cone


		accuracies_first += [float(count_first) / args.trials]
		accuracies_first_diff += [float(count_first_diff) / args.trials]
		accuracies_ml += [float(count_ml) / args.trials]
		accuracies_rc += [float(count_rc) / args.trials]
		accuracies_cone += [float(count_cone) / args.trials]
		accuracies_neighbor_cone += [float(count_neighbor_cone) / args.trials]

		if args.verbose:
			print '[Trickle] accuracies, first-spy:', accuracies_first
			# print 'accuracies, ML line:', accuracies_ml_line
			print '[Trickle] accuracies, ML:', accuracies_ml
			print '[Diffusion] accuracies, first-spy:', accuracies_first_diff
			print '[Diffusion] accuracies, distance centrality:', accuracies_cone

		if args.write:
			result_types = ['first-spy accuracy', 'ML accuracy', 'first-spy accuracy diffusion', 
							'distance centrality diffusion','cone diffusion','neighbor cone diffusion']
			param_types = ['degrees']
			results = [[accuracies_first], [accuracies_ml], [accuracies_first_diff], [accuracies_dist],[accuracies_cone],[accuracies_neighbor_cone]]
			params = [[i for i in degrees]]
			write_results(result_types, results, param_types, params, args.run)

	print 'The first-spy estimator accuracy: ', accuracies_first
	print 'The ML estimator accuracy: ', accuracies_ml
	print 'The rumor centrality estimator accuracy: ', accuracies_rc
	print 'The first-spy estimator accuracy, diffusion: ', accuracies_first_diff
	print 'The cone estimator accuracy, diffusion: ', accuracies_cone
	print 'The neighbor cone estimator accuracy, diffusion: ', accuracies_neighbor_cone
	# print 'Tested on degrees', degrees

	if args.measure_time:
		end = time.time()
		print 'The runtime is ', end-start
