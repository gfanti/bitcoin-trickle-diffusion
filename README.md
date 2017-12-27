# bitcoin-trickle-diffusion
Simulations illustrating differences between trickle and diffusion, for reproducing the plots from Fanti and Viswanath's "Deanonymization in the Bitcoin P2P Network," NIPS 2017.

The two relevant files to run are main.py (for regular tree simulations) and dataset_main.py (for simulations on a real Bitcoin graph topology). Both files take the same input arguments, which are listed below:
	"-r", "--run", type=int, help="changes the filename of saved data"
	"-v", "--verbose", help="increase output verbosity", action="store_true"
	"-w", "--write", help="writes the results to file", action="store_true"
	"-t","--trials", type=int, help="number of trials", default=1
	"-s","--spreading", type=int, help="Which spreading protocol to use (0)trickle, (1)diffusion", default=0
	"-e","--estimator", dest='estimators',default=[], type=int, help="Which estimator to use (0)first-spy, (1)ML (approximate)", action='append'


We include below instructions for running the simulations used to generate each of our figures. 

Figure 2: Run the following
python main.py -t 5000 -s 0 -w -e 0

Figure 5: 
python main.py -t 5000 -s 0 -w -e 0 
python main.py -t 5000 -s 0 -w -e 1

Figure 6: Comparison of trickle and diffusion on 4-regular trees
python main.py -t 5000 -s 0 -w -e 0 -d 4
python main.py -t 5000 -s 1 -w -e 0 -d 4

Figure 7: First-spy estimator for both diffusion and trickle, on a snapshot of the Bitcoin P2P graph from 2015
python dataset_main.py -t 5000 -s 0 -w -e 0
python dataset_main.py -t 5000 -s 1 -w -e 0

