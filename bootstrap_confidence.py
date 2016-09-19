from mdp import *
from my_birl import *


def compute_value_ratios(expert_mdp, demos, chain_length, chain_burn):
    #get mean and MAP from chain
    birl = BIRL(demos, expert_mdp.get_grid_size(), expert_mdp.terminals, expert_mdp.init,
        step_size=0.5, birl_iteration = chain_length)
    chain, map_mdp =  birl.run_birl()
    mean_reward = average_chain(chain, chain_burn)
    mean_mdp = deepcopy(map_mdp)
    mean_mdp.reward = mean_reward
    
    #print "map estimate"
    #map_mdp.print_rewards()
    #print "map policy:"
    #map_mdp.print_arrows()
    #print "expected return"
    #print expected_return(map_mdp)
    print "---------------"
    print "mean estimate"
    mean_mdp.print_rewards()
    print "mean policy:"
    print policy_iteration(mean_mdp)
    mean_mdp.print_arrows()
    print "mean expected return"
    print expected_return(mean_mdp)
    print "mean evaluated return"
    print evaluate_expected_return(best_policy(mean_mdp, value_iteration(mean_mdp, 0.001)), expert_mdp)
    
def expected_return(mdp):
    #returns the discounted expected reward from acting optimally in 
    #input mdp with initial_dist determining start state
    
    #get optimal policy and values
    pi, U = policy_iteration(mdp)
    
    #return values from starting in initial_dist
    return sum([p * U[s0] for (p, s0) in mdp.init])
    
    
#one that works for non-optimal policy (different than one induced by reward
def evaluate_expected_return(pi, eval_mdp):
    U = dict([(s, 0) for s in eval_mdp.states])
    U = policy_evaluation(pi, U, eval_mdp, k=100)
    return sum([p * U[s0] for (p, s0) in eval_mdp.init])
    
# one for demos
def evaluate_expected_return_demos(demo_set):
    for demo in demo_set:
        for s,a in demo:
            print s
    
    

