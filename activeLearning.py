#test script to understand birl implementation better
import mdp
from my_birl_batch import *
from my_birl import *
init = (1,0)
chain_length = 5000
chain_burn = 200
def main():
    expert_mdp = mdp.GridMDP([[+10, -5, -5],
                          [-1, -1, -1]],
                      terminals=[(0,1)], init = init)
                      
#print expert_mdp.R((0,1))
#print expert_mdp.T((0,0),(-1,0))
#print expert_mdp.actions((0,0))
#print expert_mdp.states
#print expert_mdp.get_grid()
#print mdp.value_iteration(expert_mdp)
#print mdp.get_q_values(expert_mdp, mdp.value_iteration(expert_mdp))

    expert_policy = best_policy(expert_mdp, value_iteration(expert_mdp, 0.001))
    print "Expert rewards:"
    expert_mdp.print_rewards()
    print "Expert policy:"
    print_table(expert_mdp.to_arrows(expert_policy))
    print "---------------"
    demo = []
    #demo.append(mdp.generate_demonstration(init, expert_policy, expert_mdp))
    #demo.append(mdp.generate_demonstration((2,0), expert_policy, expert_mdp))
    demo.append(mdp.generate_demonstration((2,1), expert_policy, expert_mdp))
    
    print demo
    
    birl = BIRL_BATCH(demo, expert_mdp.get_grid_size(), expert_mdp.terminals, expert_mdp.init,
        step_size=1.0, birl_iteration = chain_length)
    chain, mapMDP =  birl.run_birl()
    #print len(chain)
    #print chain
    mean_reward = average_chain(chain, chain_burn)
    print mean_reward
    meanMDP = deepcopy(mapMDP)
    meanMDP.reward = mean_reward
    print "---- map -----"
    mapMDP.print_rewards()
    #TODO fix print table
    mapMDP.print_arrows()
    print "---- mean -----"
    meanMDP.print_rewards()
    #TODO fix print table
    meanMDP.print_arrows()

    #TODO figure out how to print out rewards (see guys github stuff)
    #TODO plot how often it finds a better reward, not often
    #TODO plot how often it switches, also not often, NOTE switching is good since it gives coverage and support to equally good options that span the space of possible rewards
    
    print chain_variance(chain, chain_burn)
    
    #figure out the variance in the rewards in the post-burn chain
#compute the average reward over the chain of rewards
def chain_variance(chain, burn):
    mean_reward = average_chain(chain, burn)
    var_reward = {}
    count = 1.0
    #initialize
    for item in chain[burn]:
        var_reward[item] = (chain[burn][item] - mean_reward[item]) ** 2

    #add up all rewards
    for i in range(burn+1,len(chain)):
        count += 1.0
        for item in chain[i]:
            var_reward[item] += (chain[i][item] - mean_reward[item]) ** 2

    #calculate average
    for thing in mean_reward:
        var_reward[thing] = var_reward[thing] / count
    return var_reward




if __name__ == "__main__":
    main()
    
    
    
    
