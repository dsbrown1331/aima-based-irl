"""Markov Decision Processes (Chapter 17)

First we define an MDP, and the special case of a GridMDP, in which
states are laid out in a 2-dimensional grid.  We also represent a policy
as a dictionary of {state:action} pairs, and a Utility function as a
dictionary of {state:number} pairs.  We then define the value_iteration 
and policy_iteration algorithms."""

from utils import *
from random import *
import math
from scipy.misc import logsumexp


class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by 
    algorithms. The transition model is represented somewhat differently from 
    the text.  Instead of T(s, a, s') being  probability number for each 
    state/action/state triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and 
    actions for each state. [page 615]"""

    def __init__(self, init, actlist, terminals, gamma=.95):
        update(self, init=init, actlist=actlist, terminals=terminals,
               gamma=gamma, states=set(), reward={})

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        return NotImplemented

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a 
        fixed list of actions, except for terminal states. Override this 
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist



class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1].  All you have to do is 
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state).  Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""


    def __init__(self, grid, terminals, init=(0, 0), gamma=.95, r_min = -10.0, r_max = 10.0):
        grid.reverse()  ## because we want row 0 on bottom, not on top
        MDP.__init__(self, init, actlist=orientations,
                     terminals=terminals, gamma=gamma)
        self.r_min = r_min
        self.r_max = r_max
        update(self, grid=grid, rows=len(grid), cols=len(grid[0]))
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    #not sure why this is here...
    def get_grid(self):
        grid = self.grid
        print grid
        for x in range(self.cols):
            for y in range(self.rows):
                grid[y][x] = self.reward[x, y]
                print grid
        grid.reverse()
        return grid

    def print_rewards(self):
        proper_list = self.to_grid(self.reward);
        for x in (range(self.rows)):
            for y in (range(self.cols)):
                print("%.2f" % round(proper_list[x][y], 2)),
            print(" ")


    def get_grid_size(self):
        return len(self.grid), len(self.grid[0])


    def modify_rewards_randomly(self, step=0.05):
        x_to_change = randint(0, self.cols - 1)
        y_to_change = randint(0, self.rows - 1)
        self.modify_state((x_to_change,y_to_change), step)

    def get_max_reward(self): return max(self.reward.values())

    def get_min_reward(self): return min(self.reward.values())

    def scale_true_reward(self):
    # This is necessary if we have no information about the prior information
    # scales the true reward functions so we have a better measure of reward
    # loss

        for key in self.reward:
            self.reward[key] /= max(self.get_max_reward(), abs(self.get_min_reward()))

        diff = [BIG_NUMBER, BIG_NUMBER]
        # if any(true_reward < 0):
        diff[0] = self.r_min / (self.get_min_reward() - SMALL_NUMBER)

        # if any(true_reward > 0):
        diff[1] = self.r_max / (self.get_max_reward() + SMALL_NUMBER)

        for key in self.reward:
            self.reward[key] *= min(diff)


    # def T(self, state, action):
    #     if action == None:
    #         return [(0.0, state)]
    #     else:
    #         return [(0.8, self.go(state, action)),
    #                 (0.1, self.go(state, turn_right(action))),
    #                 (0.1, self.go(state, turn_left(action)))]

    def T(self, state, action):
        if action == None:
            return [(0.0, state)]
        else:
            return [(1.0, self.go(state, action))]

    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = vector_add(state, direction)
        return if_(state1 in self.states, state1, state)

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x, y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0): '>', (0, 1): '^', (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))

    def print_arrows(self):
        print_table(self.to_arrows(best_policy(self, value_iteration(self, 0.001))))

    def modify_state(self, indices, step):
        #y_to_change, x_to_change = indices #hopefully this was an error!!
        x_to_change, y_to_change = indices
        direction = randint(0, 1) * 2 - 1
        if self.r_min < self.reward[x_to_change, y_to_change] + direction * step < self.r_max:
            self.reward[x_to_change, y_to_change] += direction * step



#___________________________________________________________________
class DeterministicGridMDP(GridMDP):
    """A two-d grid MDP with deterministic transition dynamics"""
    
    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        else:
            return [(1.0, self.go(state,action))]


#___________________________________________________________________
class DeterministicWeightGridMDP(DeterministicGridMDP):
    """ A two-d grid MDP with weights determining rewards
    weights are a dictionary mapping features to rewards
    grid is a 2d array of features that match weight dict
    """
    def __init__(self, features, weights, grid, terminals, init=[(0, 0)], gamma=.9):
        grid.reverse()  # because we want row 0 on bottom, not on top
        MDP.__init__(self, init, actlist=orientations,
                     terminals=terminals, gamma=gamma)
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.features = features
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = weights[grid[y][x]]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def observe_features(self, state):
        x,y = state
        return self.grid[y][x]


#______________________________________________________________________________


def value_iteration(mdp, epsilon=0.001):
    "Solving an MDP by value iteration. [Fig. 17.4]"
    U1 = dict([(s, 0) for s in mdp.states])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        #print U1
        if delta < epsilon * (1 - gamma) / gamma:
            return U


def get_q_values(mdp, U):
    Q = {}
    for s in mdp.states:
        for a in mdp.actions(s):
            Qtemp = mdp.reward[s]
            for (p, sp) in mdp.T(s, a):
                Qtemp += mdp.gamma * p * U[sp]
            Q[s, a] = Qtemp
    return Q



def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), lambda a: expected_utility(a, s, U, mdp))
    return pi


def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])


#______________________________________________________________________________

def policy_iteration(mdp):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    U = dict([(s, 0) for s in mdp.states])
    pi = dict([(s, choice(mdp.actions(s))) for s in mdp.states])
    while True:
        U = policy_evaluation(pi, U, mdp)
        #print "policy eval U"
        #print U
        #print "policy eval PI"
        #print pi
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s), lambda a: expected_utility(a, s, U, mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi, U

#TODO what is a good value for k?
def policy_evaluation(pi, U, mdp, k=100):
    """Return an updated utility mapping U from each state in the MDP to its 
    utility, using an approximation (modified policy iteration)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        #print "k", i
        for s in mdp.states:
            #print "s", s
            U[s] = R(s) + gamma * sum([p * U[s1] for (p, s1) in T(s, pi[s])])
            #print "U[s]", U[s]
    return U

#how to generate a demo from a start location
###TODO  this is for deterministic settings!!
###TODO add a terminationg criterion like value and policy iteration!
def generate_demonstration(start, policy, mdp):
    """given a start location return the demonstration following policy
    return a state action pair array"""
    
    demonstration = []
    curr_state = start
    #print('start',curr_state)
    
    while curr_state not in mdp.terminals:
        #print('action',policy[curr_state])
        demonstration.append((curr_state, policy[curr_state]))
        curr_state = mdp.go(curr_state, policy[curr_state])
        #print('next state', curr_state)
    #append the terminal state
    demonstration.append((curr_state, None))
    return demonstration


