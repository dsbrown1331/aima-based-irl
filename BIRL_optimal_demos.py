import numpy as np
import mdp
from my_birl_batch import *
from my_birl import *
from halfspace_uncertainty import *
from mdp_feature_counts import *
from optimal_teaching import *
from activeLearning import chain_variance
import operator

for size in range(3,4):
    print "^^^^^^", size, "^^^^^^^"
    f = open('active_results/optimalTest' + str(size)+ '2.txt','w')
    for iter in range(10):
        print "-----", iter, "------"
        #generate a random n by n world
        grid_width = size
        grid_height = size
        rand_reward = []
        for row in range(grid_height):
            temp = []
            for col in range(grid_width):
                temp.append(np.random.randint(-10,0))
            rand_reward.append(temp)
        rand_reward[0][0] = 10

        ###for debugging
        #rand_reward = [[10.00, -5.00, -5.00],  
        #[-1.00, -1.00, -1.00 ]]
        ###

        terminals=[(0,grid_height-1)]
        init = []
        for row in range(grid_height):
            for col in range(grid_width):
                if row == grid_height-1 and col == 0:
                    continue
                init.append((col,row))
        print "init"
        print init


        expert_mdp = mdp.GridMDP(deepcopy(rand_reward), terminals, init)
        expert_mdp.print_rewards()
        expert_mdp.print_arrows()



        #try Cakmak's Task 1 with just one start to see if it gets the same demo
        #birlToy = DeterministicWeightGridMDP(
        #    features = ['f0', 'f1', 'f2'],
        #    weights = {'f0': 10, 'f1': -5, 'f2': -1, None: None},
        #    grid = [['f0', 'f1', 'f1'],
        #            ['f2', 'f2', 'f2']],
        #    terminals=[(0,1)],
        #    init = [(0,0),(1,0),(1,1),(2,0),(2,1)], gamma = 0.9)
        features = []
        count = 0
        for row in range(grid_height):
            for col in range(grid_width):
                features.append('f' + str(count))
                count += 1
        #print "features"
        #print features

        weights = {}
        count = 0
        for row in range(grid_height):
            for col in range(grid_width):
                #print row,col
                weights[features[count]] = rand_reward[row][col]
                count += 1
        weights[None] = None
        print "weights"
        print weights

        grid = []
        count = 0
        for row in range(grid_height):
            temp = []
            for col in range(grid_width):
                temp.append(features[count])
                count += 1        
            grid.append(temp)
        #print "grid"
        #print grid

        #select random init state
        demo_init = init[np.random.randint(0,len(init))]
        print "demo_init"
        print demo_init
        #generate random demo

        demo = []
        expert_policy = best_policy(expert_mdp, value_iteration(expert_mdp, 0.001))
        demo.append(mdp.generate_demonstration(demo_init, expert_policy, expert_mdp))
        print "demo"
        print demo

        rand_task = DeterministicWeightGridMDP(
            features, weights, grid, terminals, init, gamma = 0.95)
        #rand_task.print_rewards()
        #rand_task.print_arrows()
        cakmak_optimal = seeded_optimal_teaching(demo,rand_task, 100000,10)
        #print("solution: ", cakmak_optimal)
        score, cakmak_demo = cakmak_optimal
        cakmak_init = cakmak_demo[0][0]
        print "cakmak", cakmak_init

        #compare to BIRL active learning reward variance approach
        chain_length = 12000
        chain_burn = 2000
        birl = BIRL_BATCH(demo, expert_mdp.get_grid_size(), expert_mdp.terminals, expert_mdp.init,
            step_size=1.0, birl_iteration = chain_length)
        chain, mapMDP =  birl.run_birl()

        chain_var =  chain_variance(chain, chain_burn)
        #find highest variance that's not start of demo or terminal state
        chain_var.pop(terminals[0])
        sorted_var = sorted(chain_var.items(), key=operator.itemgetter(1))
        sorted_var.reverse()      
        query_states = [state for state, var in sorted_var]
        print query_states
        indx = query_states.index(cakmak_init)
        print indx


        f.write(str(indx) + '\n') # python will convert \n to os.linesep
    f.close()

        



