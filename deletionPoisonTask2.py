#optimal teaching algorithm from Cakmak paper
from halfspace_uncertainty import *
from mdp_feature_counts import *
from optimal_teaching import *


print("________________________________________")
print("Task 2")
print("________________________________________")


#try Cakmak's Task 2 with just one start to see if it gets the same demo
task2 = DeterministicWeightGridMDP(
    features = ['f0', 'f1', 'f2'],
    weights = {'f0': 2, 'f1': -1, 'f2': -10, None: None},
    grid = [[None, 'f0', None, None, 'f1', None, None],
            [None, 'f1', 'f1', 'f1', 'f1', 'f1', None],
            [None, 'f2', None, None, None, 'f1', None],
            [None, 'f2', None, None, None, 'f1', None],
            [None, 'f1', 'f1', 'f1', 'f1', 'f1', None],
            ['f1', 'f1', None, None, None, None, None]],
    terminals=[(1,5)],
    init = [(0,0),(1,0),(1,1),(1,2),(1,3),(1,4),
        (2,1),(3,1),(4,1),(5,1),(5,2),(5,3),(5,4),
        (1,4),(2,4),(3,4),(4,4),(4,5)], gamma = 0.9)

#task2 = GridMDP(
#    grid = [[None, 2, None, None, -1, None, None],
#            [None, -1, -1, -1, -1, -1, None],
#            [None, -10, None, None, None, -1, None],
#            [None, -10, None, None, None, -1, None],
#            [None, -1, -1, -1, -1, -1, None],
#            [-1, -1, None, None, None, None, None]],
#    terminals=[(1,5)],
#    init = [(0,0),(1,0),(1,1),(1,2),(1,3),(1,4),
#        (2,1),(3,1),(4,1),(5,1),(5,2),(5,3),(5,4),
#        (1,4),(2,4),(3,4),(4,4),(4,5)], gamma = 0.9)


maxG, bestDemo = stochastic_optimal_teaching(task2, 100000, 1)
print("optimal")
print(maxG)
print(bestDemo)
print(len(bestDemo))

#remove first element from bestDemo and recalculate uncertainty
Gp = []
Gp.append(maxG) #for not removing anything
poisoned = list(bestDemo)
for d in range(len(bestDemo)-2): #can't remove second to last one since there is nothing to do with the last terminal element
    print("poisoning by removing %dth element" % d)
    popped = poisoned.pop(0)
    print(popped)
    print(poisoned)
    G = evalutate_uncertainty_traj(poisoned, task2, 100000, 1)
    print(G)
    Gp.append(G)

print(Gp)
