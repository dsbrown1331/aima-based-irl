from mdp import *
from my_birl import *
from my_birl_batch import *

def compute_value_ratios(expert_mdp, demos, chain_length, chain_burn):
    #get mean and MAP from chain
    birl_batch = BIRL_BATCH(demos, expert_mdp.get_grid_size(), expert_mdp.terminals, expert_mdp.init,
        step_size=0.5, birl_iteration = chain_length)
    chain, map_mdp = birl_batch.run_birl()
    mean_reward = average_chain(chain, chain_burn)
    mean_mdp = deepcopy(map_mdp)
    mean_mdp.reward = mean_reward
    
    print "map estimate"
    map_mdp.print_rewards()
    print "map policy:"
    map_mdp.print_arrows()
    print "expected return"
    print expected_return(map_mdp)
    
    print "---------------"
    print "mean estimate2"
    mean_mdp.print_rewards()
    print "mean policy2:"
    print policy_iteration(mean_mdp)
    mean_mdp.print_arrows()
    print "mean expected return2"
    print expected_return(mean_mdp)
    print "mean evaluated return2"
    print evaluate_expected_return(best_policy(mean_mdp, value_iteration(mean_mdp, 0.001)), expert_mdp)
    
    print "------------"
    print "V^demo(R*)", evaluate_expected_return_demos(demos, expert_mdp)
    print "(Vdemo - Vpi_mean)/Vdemo", policy_value_ratio(mean_reward, expert_mdp.reward, demos, expert_mdp)
    print "(Vdemo - Vpi_map)/Vdemo", policy_value_ratio(map_mdp.reward, expert_mdp.reward, demos, expert_mdp)
    
    
#takes an estimated reward, demos, an MDP\R and evaluates the ratio of values if reward_eval is true reward
##TODO note this assumes the same start state for mdp\r and for demos!!
def policy_value_ratio(reward_est, reward_eval, demos, mdp_r):
    #print "reward_est", reward_est
    #print "reward_eval", reward_eval
    #first get optimal policy if reward_est is true
    mdp_est = deepcopy(mdp_r)
    mdp_est.reward = reward_est
    pi_est, U_est = policy_iteration(mdp_est)
    #print "pi_est", pi_est
    #evaluate pi_est and demos on MDP\R + reward_eval
    mdp_eval = deepcopy(mdp_r)
    mdp_eval.reward = reward_eval
    V_pi_est = evaluate_expected_return(pi_est, mdp_eval)
    V_demos = evaluate_expected_return_demos(demos, mdp_eval)
    #print "V_pi_est", V_pi_est
    #print "V_demos", V_demos
    #calculate the ratio
    ratio = (V_demos - V_pi_est) / V_demos
    return ratio

#function to get BIRL mean policy
#def 
    
def expected_return(mdp):
    #returns the discounted expected reward from acting optimally in 
    #input mdp with initial_dist determining start state
    
    #get optimal policy and values
    pi, U = policy_iteration(mdp)
    
    #return values from starting in initial_dist
    return sum([p * U[s0] for (p, s0) in mdp.init])
    
    
#one that works for non-optimal policy (different than one induced by reward
def evaluate_expected_return(pi, eval_mdp):
    #print "eval mdp reward", eval_mdp.reward
    #print "evaluating at states", eval_mdp.init
    U = dict([(s, 0) for s in eval_mdp.states])
    U = policy_evaluation(pi, U, eval_mdp, k=100)
    #print "U", U
    return sum([p * U[s0] for (p, s0) in eval_mdp.init])

#TODO test this out 
# one for demos
def evaluate_expected_return_demos(demo_set, eval_mdp):
    #print "evaluating demos"
    expected_reward = 0.0
    for demo in demo_set:
        t = 0
        for s,a in demo:
            #print "reward", eval_mdp.R(s), " for state", s
            expected_reward += (eval_mdp.gamma ** t)  * eval_mdp.R(s)
            #print expected_reward
            t += 1
    return expected_reward / len(demo_set)
    

