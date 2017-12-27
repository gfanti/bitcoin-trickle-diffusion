# bitcoin-trickle-diffusion
Simulations illustrating differences between trickle and diffusion, for reproducing the plots from Fanti and Viswanath's "Deanonymization in the Bitcoin P2P Network," NIPS 2017.

The two relevant files to run are main.py (for regular tree simulations) and dataset_main.py (for simulations on a real Bitcoin graph topology). Both files take the same input arguments, which are listed below:

	```
	"-r", "--run", type=int, help="changes the filename of saved data"
	"-v", "--verbose", help="increase output verbosity", action="store_true"
	"-w", "--write", help="writes the results to file", action="store_true"
	"-t","--trials", type=int, help="number of trials", default=1
	"-s","--spreading", type=int, help="Which spreading protocol to use (0)trickle, (1)diffusion", default=0
	"-e","--estimator", dest='estimators',default=[], type=int, help="Which estimator to use (0)first-spy, (1)ML (approximate)", action='append'
	"-d", "--degree", type=int, help="fixed degree of tree", default=0  <-- if you don't specify this, the code just runs an array of degrees for regular 
			trees. In dataset_main, this argument is ignored, since the graph is fixed
	```


We include below instructions for running the simulations in our paper. You may need to tune the runtimes (especially for diffusion) if the simulations are taking too long. This shouldn't affect accuracy too much, provided you run the simulation environment long enough for a node's local environment to get the message/transaction. 

Figure 2: First-timestamp estimator accuracy on d-regular trees when theta = 1
python main.py -t 5000 -s 0 -w -e 0 (OK)

Figure 5: First-timestamp vs. reporting centrality on diffusion over regular trees; theta=1
python main.py -t 5000 -s 0 -w -e 1  (NOT WORKING)

Figure 6: Comparison of trickle and diffusion on 4-regular trees, sweep theta
python main.py -t 5000 -s 0 -w -e 0 -d 4 -q (OK)
python main.py -t 5000 -s 1 -w -e 0 -d 4 -q (OK)

Figure 7: First-spy estimator for both diffusion and trickle, on a snapshot of the Bitcoin P2P graph from 2015
python dataset_main.py -t 5000 -s 0 -w -e 0 (OK)
python dataset_main.py -t 5000 -s 1 -w -e 0 (OK)

