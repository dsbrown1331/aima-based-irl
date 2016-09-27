#optimal teaching algorithm from Cakmak paper
from halfspace_uncertainty import *
from mdp_feature_counts import *
def evaluate_uncertainty(start, mdp, num_samples, Mw):
    """given a starting location return corresponding uncertanity, demonstration pair
    """
    opt_policy = best_policy(mdp, value_iteration(mdp))
    opt_demo = generate_demonstration(start, opt_policy, mdp)
    ##print("opt_demo",opt_demo)
    all_constraints = generate_all_constraints(opt_demo,opt_policy,mdp)
    ##print("all_constraints",all_constraints)
    GofD = calculate_uncertainty(all_constraints, num_samples, Mw)
    ##print("uncertainty", GofD)
    return GofD, opt_demo

def evalutate_uncertainty_traj(opt_demo,mdp, num_samples, Mw):
    opt_policy = best_policy(mdp, value_iteration(mdp))
    all_constraints = generate_all_constraints(opt_demo,opt_policy,mdp)
    ##print("all_constraints",all_constraints)
    non_duplicated_constraints = list(set([tuple(c) for c in all_constraints]))
    ##print("non-duplicates", non_duplicated_constraints)
    GofD = calculate_uncertainty(non_duplicated_constraints, num_samples, Mw)
    ##print("uncertainty", GofD)
    return GofD
    
    
def evalutate_uncertainty_traj_seeded(init_demo,opt_demo,mdp, num_samples, Mw):
    opt_policy = best_policy(mdp, value_iteration(mdp))
    seed_constraints = generate_all_constraints(init_demo,opt_policy,mdp) 
    all_constraints = generate_all_constraints(opt_demo,opt_policy,mdp)
    all_constraints.extend(seed_constraints)
    ##print("all_constraints",all_constraints)
    non_duplicated_constraints = list(set([tuple(c) for c in all_constraints]))
    ##print("non-duplicates", non_duplicated_constraints)
    GofD = calculate_uncertainty(non_duplicated_constraints, num_samples, Mw)
    ##print("uncertainty", GofD)
    return GofD
    
    
#need a way to get all possible optimal trajectories that start at a certain point
def all_optimal_trajectories(start, mdp):
    values = value_iteration(mdp)
    #find values of movements
    maxVal = -float('inf')
    optimal_trajs = []
    for a in mdp.actions(start):
        if(values[mdp.go(start,a)] > maxVal):
            maxVal = values[mdp.go(start,a)]
    for a in mdp.actions(start):
        if(values[mdp.go(start,a)] == maxVal):
            traj = []
            traj.append((start,a))
            take_best_step(mdp.go(start,a),traj, mdp, values, optimal_trajs)
    return optimal_trajs

def take_best_step(next, traj, mdp, values, optimal_trajs):
    if next in mdp.terminals:
        traj.append((next,None))
        optimal_trajs.append(traj)
        return 
    else:
        maxVal = -float('inf')
        for a in mdp.actions(next):
            if(values[mdp.go(next,a)] > maxVal):
                maxVal = values[mdp.go(next,a)]
        for a in mdp.actions(next):
            if(values[mdp.go(next,a)] == maxVal):
                traj.append((next,a))
                take_best_step(mdp.go(next,a),traj, mdp, values, optimal_trajs)
    
    
def deterministic_optimal_teaching(mdp, num_samples = 200000, Mw = 1):
    """run the Cakmak algorithm for a deterministic MDP with a single
    trajectory demo"""

    #evalute each start site and find best single demonstration
    maxG = -float('inf')
    bestDemo = []
    for start in mdp.init:
	##print("evaluating start", start)
        Gd,demo = evaluate_uncertainty(start, mdp, num_samples, Mw)
        ##print('start',start,'Gd',Gd,'demo',demo)
        if Gd > maxG:
            maxG = Gd
            bestDemo = demo
    return maxG, bestDemo        
    

def stochastic_optimal_teaching(mdp, num_samples = 20000, Mw = 10):
    """run the Cakmak algorithm for a deterministic MDP but with
    a stochastic opitimal policy """

    #evalute each start site and find best single demonstration
    maxG = -float('inf')
    bestDemo = []
    for start in mdp.init:
        for alt_traj in all_optimal_trajectories(start, mdp):
            #print("evaluating start", start)
            #print("with traj", alt_traj)
            Gd = evalutate_uncertainty_traj(alt_traj, mdp, num_samples, Mw)
            ##print('start',start,'Gd',Gd,'demo',alt_traj)
            if Gd > maxG:
                maxG = Gd
                bestDemo = alt_traj
    return maxG, bestDemo 

#uses a given initial demo
def seeded_optimal_teaching(init_demo, mdp, num_samples = 20000, Mw = 10):
    """run the Cakmak algorithm for a deterministic MDP but with
    a stochastic opitimal policy """
    
   
    #evalute each start site and find best single demonstration
    maxG = -float('inf')
    bestDemo = []
    for start in mdp.init:
        for alt_traj in all_optimal_trajectories(start, mdp):
            #print("evaluating start", start)
            #print("with traj", alt_traj)
            Gd = evalutate_uncertainty_traj_seeded(init_demo, alt_traj, mdp, num_samples, Mw)
            ##print('start',start,'Gd',Gd,'demo',alt_traj)
            if Gd > maxG:
                maxG = Gd
                bestDemo = alt_traj
    return maxG, bestDemo 

