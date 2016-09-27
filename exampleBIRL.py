import numpy as np
import mdp
from my_birl_batch import *
from my_birl import *
from halfspace_uncertainty import *
from mdp_feature_counts import *
from optimal_teaching import *
from activeLearning import chain_variance
import operator


#generate a random n by n world
grid_width = 3
grid_height = 2
rand_reward = []
for row in range(grid_height):
    temp = []
    for col in range(grid_width):
        temp.append(np.random.randint(-10,0))
    rand_reward.append(temp)
rand_reward[0][0] = 10

###for debugging
rand_reward = [[10.00, -5.00, -5.00],  
[-1.00, -1.00, -1.00 ]]
###

terminals=[(0,grid_height-1)]
init = []



expert_mdp = mdp.GridMDP(deepcopy(rand_reward), terminals, init)
expert_mdp.print_rewards()
expert_mdp.print_arrows()



#select random init state
demo_init = (2,1)
print "demo_init"
print demo_init
#generate random demo

demo = []
expert_policy = best_policy(expert_mdp, value_iteration(expert_mdp, 0.001))
demo.append(mdp.generate_demonstration(demo_init, expert_policy, expert_mdp))
print "demo"
print demo


#compare to BIRL active learning reward variance approach
chain_length = 5000
chain_burn = 200
birl = BIRL_BATCH(demo, expert_mdp.get_grid_size(), expert_mdp.terminals, expert_mdp.init,
    step_size=1.0, birl_iteration = chain_length)
chain, mapMDP =  birl.run_birl()

mapMDP.print_rewards()
mapMDP.print_arrows()







