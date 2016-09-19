from mdp import *
from my_birl import *


def compute_value_ratios(expert_mdp, demos, chain_length, chain_burn):
    #get mean and MAP from chain
    birl = BIRL(demos, expert_mdp.get_grid_size(), expert_mdp.terminals, 
        step_size=0.5, birl_iteration = chain_length)
    chain, map_mdp =  birl.run_birl()
    mean_reward = average_chain(chain, chain_burn)
    mean_mdp = deepcopy(map_mdp)
    mean_mdp.reward = mean_reward
    
    #print "map estimate"
    #map_mdp.print_rewards()
    #print "map policy:"
    #map_mdp.print_arrows()
    #print "---------------"
    #print "mean estimate"
    #mean_mdp.print_rewards()
    #print "mean policy:"
    #mean_mdp.print_arrows()
    
#TODO test
def expected_return(mdp, initial_dist):
    #returns the discounted expected reward from acting optimally in 
    #input mdp with initial_dist determining start state
    
    #get optimal policy and values
    pi, U = policy_iteration(mdp)
    
    #return values from starting in initial_dist
    return sum([p * U[s0] for (p, s0) in initial_dist])
    
    
#TODO write one that works for non-optimal policy (different than one induced by reward

#TODO write one for demos
