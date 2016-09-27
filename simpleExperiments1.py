import mdp
from my_birl import *
from bootstrap_confidence import *

def main():
    print "experiment on simple 2x3 grid world"
    chain_length = 3000
    chain_burn = 1000
    terminals =[(0,2)] 
    init_dist = [(0.5,(2,0)), (0.5,(1,2))]
    #true_reward = [[+10, -5, -5],
    #               [-1, -1, -1]]
    
    true_reward = [[+10, -10, 0],
                   [  0, -10, 0],
                   [  0,   0, 0]]
    
    expert_mdp = mdp.GridMDP(true_reward,
                      terminals, init_dist, gamma=.95)
    expert_mdp.print_arrows()
    #get mean and MAP from chain
    #not sure how expert demos should be given, I think we want a start and end state
    print "policy iteration"
    expert_policy, true_U = mdp.policy_iteration(expert_mdp)
    expert_demo = []
    print "generating demo"
    expert_demo.append(mdp.generate_demonstration((2,0), expert_policy, expert_mdp))
    expert_demo.append(mdp.generate_demonstration((1,2), expert_policy, expert_mdp))
    #expert_demo.append(mdp.generate_demonstration((2,0), expert_policy, expert_mdp))
    #expert_demo.append(mdp.generate_demonstration((1,0), expert_policy, expert_mdp))
    #expert_demo.append(mdp.generate_demonstration((0,0), expert_policy, expert_mdp))
    #expert_demo.append([((0, 1), None), ((0, 0), (0, 1)), ((2, 1), (0, -1)), ((2, 0), (-1, 0)), ((1, 0), (-1, 0)), ((1, 1), (-1, 0))])
    #expert_demo = expert_policy
    print "demo"
    print expert_demo
    print "True rewards:"
    expert_mdp.print_rewards()
    print "Expert policy:"
    print_table(expert_mdp.to_arrows(expert_policy))
    print "mean expected return"
    print expected_return(expert_mdp)
    print value_iteration(expert_mdp, 0.001)
    print "---------------"
    
    compute_value_ratios(expert_mdp, expert_demo, chain_length, chain_burn)














if __name__ == "__main__":
    main()
