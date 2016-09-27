"""
Daniel Brown
based on aima-based-irl
"""

from mdp import *
from utils import *
from copy import deepcopy
from math import exp
from my_birl import BIRL

#takes 
class BIRL_BATCH(BIRL):
    def __init__(self, expert_trace, grid_size, terminals, init, step_size=1.0, r_min=-10.0,
                 r_max=10.0, prior = 'uniform', birl_iteration = 2000):
        self.n_rows, self.n_columns = grid_size
        self.r_min, self.r_max = r_min, r_max
        self.step_size = step_size
        print "step size", self.step_size
        print "r_min", r_min
        print "r_max", r_max
        self.expert_trace = merge_trajectories(expert_trace)
        print 'trace for batch', self.expert_trace
        self.terminals = terminals
        self.init = init
        self.prior = prior #string to specify what type of prior
        self.birl_iteration = birl_iteration #how long to run the markov chain
       
    def run_birl(self):
        #This is the core BIRL algorithm
        Rchain = [] #store rewards along the way, #TODO dictionaries are probably not best...
        #TODO make this a random reward vector
        mdp = self.create_zero_rewards() #pick a starting reward vector
        #Rchain.append(mdp.reward) #I don't think i want the initital random reward
        
        #print 'old rewards'
        #mdp.print_rewards()
        pi, u = policy_iteration(mdp) #calculate optimal policy and utility for random R
        q = get_q_values(mdp, u) #get the q-values for R in mdp
        #print "qqqqqqq"
        #print q
        #print "qqqqqq"
        posterior = calculate_posterior(mdp, q, self.expert_trace, self.prior)
        bestPosterior = posterior 
        bestMDP = mdp
        for i in range(self.birl_iteration):
            #print "===== iter", i, "======" 
            new_mdp = deepcopy(mdp)
            new_mdp.modify_rewards_randomly(step = self.step_size) #pick random reward along grid
            #print 'new rewards'
            #new_mdp.print_rewards()
            #TODO this isn't exactly like the paper...
            #TODO ask scott about Q, should it be based on pi or pi^*
            new_u = policy_evaluation(pi, u, new_mdp) #evaluate old policy with old u and update using new R #changed to use default k which i changed to be 100
            #i wonder if it helps to start with old u? I guess a lot won't change 
            #check if there is a state where new action is better than old policy
            if pi != best_policy(new_mdp, new_u):
            #also try 
            #if q != get_q_values(new_mdp, new_u): #I think it is the same...
                #print 'old policy not optimal under new reward'
                new_pi, new_u = policy_iteration(new_mdp) #get new policy #TODO i think we could use the best_policy command above to speed things up
                new_q = get_q_values(new_mdp, new_u) #get new q-vals to calc posterior
                new_posterior = calculate_posterior(new_mdp, new_q, self.expert_trace, self.prior)
                #print 'prob switch', exp(new_posterior - posterior)
                if probability(min(1, exp(new_posterior - posterior))):
                    #I figure out why it doesn't switch very often, there was a bug in policy evaluation
                    #TODO I could do much better I think using simulated annealing or randomized hill climbing...just systemtatically try neighboring rewards and climb with some noise...isn't that what mcmc does, though, I don't know if it's all that efficient...maybe a better prior is needed
                    #TODO why does switching happen that results in worse policy is it just the weighting by Qvals?
                    #print "===== iter", i, "======" 
                    #print 'switched better'
                    #print 'new rewards'
                    #new_mdp.print_rewards()
                    #new_mdp.print_arrows()
                    #try saving the best so far
                    if bestPosterior < new_posterior:
                        bestPosterior = new_posterior
                        bestMDP = new_mdp
                        #print "best", i
                        #bestMDP.print_rewards()
                        #bestMDP.print_arrows()
                    
                    pi, u, mdp, posterior = new_pi, new_u, deepcopy(new_mdp), new_posterior

            else:
                new_q = get_q_values(new_mdp, new_u)
                new_posterior = calculate_posterior(new_mdp, new_q, self.expert_trace, self.prior)

                if probability(min(1, exp(new_posterior - posterior))):
                    #print "===== iter", i, "======" 
                    #print 'switched random'
                    #print 'new rewards'
                    #new_mdp.print_rewards()
                    #new_mdp.print_arrows()
                    mdp, posterior = deepcopy(new_mdp), new_posterior

            #print "iter", i
            #mdp.print_rewards()
            #print "---"

            Rchain.append(mdp.reward)
        return Rchain, bestMDP
        
        #------------- Reward functions ------------

    def create_zero_rewards(self):
        return GridMDP([[0 for _ in range(self.n_columns)] for _ in range(self.n_rows)]
                       , terminals=deepcopy(self.terminals), init = deepcopy(self.init),
                       r_min = self.r_min, r_max = self.r_max)

#seems correct, gives the log (P(demo | R) * P(R))
#TODO what do you do for the terminal state? when actions are None what do you normalize by
#TODO does the agent even know the terminals? what is the action in the terminal state?
#TODO anneal the alpha ?
#TODO I think I need to reverse the order of the iteration need to think about it more...
#overriden method
def calculate_posterior(mdp, q, expert_demos, prior, alpha=0.95,):
    z = []
    e = 0
    
    for s_e, a_e in expert_demos:
        #print s_e, a_e
        for a in mdp.actions(s_e):
            #print q[s_e, a]
            z.append(alpha * q[s_e, a]) #normalizing constant in denominator
        #print q[s_e,a_e]
        e += alpha * q[s_e, a_e] - logsumexp(z) #log(e^(alpha * Q) / sum e^Q)
        #print e
        
        del z[:]  #Removes contents of Z
    #TODO get a better prior and maybe use state info, not just raw values??
    if prior is 'uniform':
        return e #priors will cancel in ratio #TODO figure out how to do uniform?
    # return P(demo | R) * P(R) in log space


    
#take a bunch of lists of (s,a) tuples merge all into one list
def merge_trajectories(demos):
    merged = []  
    for demo in demos:
        merged.extend(demo)  
    return merged
#demos = [[((2, 1), (0, -1)), ((2, 0), (-1, 0)), ((1, 0), (-1, 0)), ((0, 0), (0, 1)), ((0, 1), None)], [((1, 1), (-1, 0)), ((0, 1), None)]]

#print merge_trajectories(demos)

