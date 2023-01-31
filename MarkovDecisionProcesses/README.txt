This folder contains:

 * README.txt - This file as an overview of folder contents.
 * mdp_a4- Main file that generates all plots for Markov Decision Process (MDP) Problems and model-based/model-free Algorithms :
	(1) Experiment 1: Value Iteration
	(2) Experiment 2: Policy Iteration
	(3) Experiment 3: QLearning

The two MDPs that were chosen:
	(1) FrozenLake-4x4 (Small-16 states)
	(2) Forest Management (Large- 1000 states)

The three algorithms that were used:
	(1) Value Iteration
	(2) Policy Iteration
	(3) QLearning

 * mdp.py - Modified package from hiive/mdptoolbox for Python-based MDP/RL
 * openai.py - Modified package from hiive/mdptoolbox for converting Gym MDP problems to mdptoolbox compatible


├── README.txt
├── mdp_a4.py
├── mdp.py
└── openai.py


SUMMARY OF HOW TO RUN THE CODE
=======================

To run the code in this folder, first ensure that you are using Python3.6 or greater.

Installation steps:
- Hiive/MDPToolbox: https://pypi.org/project/mdptoolbox-hiive/
- Gym: https://gym.openai.com/docs/

You can perform Experiments 1-4 for the two MDPs listed by running the following command:

python mdp_a4.py

Be sure to modify the main function based on which model you are interested in running. Some tips below.

To run all of the algorithms for a given MDP problem, do:

forest(), or
frozen_lake()

To run a specific algorithm for a given MDP problem, see the following:


===FOR ALL EXPERIMENTS===

* To generate hypertuning plots, invoke 
<ALGORITHM>_<MDP>()

e.g. value_iter_forest()

==FOR ALGORITHM COMPARISON==

* To compare across algorithms, invoke:
compare_<MDP>

e.g. compare_frozen_lake()


REFERENCES:

* Hiive/MDPToolbox: https://pypi.org/project/mdptoolbox-hiive/

* Gym: https://gym.openai.com/docs/

* FrozenLake Demo: https://gym.openai.com/envs/FrozenLake-v0/

* Forest Management Example: https://pymdptoolbox.readthedocs.io/en/latest/api/example.html

* MDPToolbox Algorithm Documentation: https://pymdptoolbox.readthedocs.io/en/latest/api/mdp.html
