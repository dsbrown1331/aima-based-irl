import mdp
from my_birl import *
from bootstrap_confidence import *

def main():
    print "experiment on simple 2x3 grid world"
    chain_length = 4000
    chain_burn = 1000
    terminals =[(0,1)] 
    expert_mdp = mdp.GridMDP([[+10, -5, -5],
                          [-1, -1, -1]],
                      terminals)
    #get mean and MAP from chain
    expert_trace = best_policy(expert_mdp, value_iteration(expert_mdp, 0.001))
    print "True rewards:"
    expert_mdp.print_rewards()
    print "Expert policy:"
    print_table(expert_mdp.to_arrows(expert_trace))
    print "---------------"

    compute_value_ratios(expert_mdp, expert_trace, chain_length, chain_burn)














if __name__ == "__main__":
    main()
