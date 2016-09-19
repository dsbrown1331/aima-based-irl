#test script to understand birl implementation better
import mdp
from my_birl import *

def main():
    expert_mdp = mdp.GridMDP([[+10, -5, -5],
                          [-1, -1, -1]],
                      terminals=[(0,1)])
                      
#print expert_mdp.R((0,1))
#print expert_mdp.T((0,0),(-1,0))
#print expert_mdp.actions((0,0))
#print expert_mdp.states
#print expert_mdp.get_grid()
#print mdp.value_iteration(expert_mdp)
#print mdp.get_q_values(expert_mdp, mdp.value_iteration(expert_mdp))

    expert_trace = best_policy(expert_mdp, value_iteration(expert_mdp, 0.001))
    print "Expert rewards:"
    expert_mdp.print_rewards()
    print "Expert policy:"
    print_table(expert_mdp.to_arrows(expert_trace))
    print "---------------"

    birl = BIRL(expert_trace, expert_mdp.get_grid_size(), expert_mdp.terminals, 
        step_size=0.5, birl_iteration = 4000)
    chain, bestMDP =  birl.run_birl()
    print len(chain)
    print average_chain(chain, 1000)
    print bestMDP.reward
    #TODO fix print table
    #print print_table(bestMDP.to_arrows(policy_iteration(bestMDP)))
    #TODO figure out how to print out rewards (see guys github stuff)
    #TODO plot how often it finds a better reward, not often
    #TODO plot how often it switches, also not often, NOTE switching is good since it gives coverage and support to equally good options that span the space of possible rewards
    




if __name__ == "__main__":
    main()
