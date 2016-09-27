from mdp import *
import numpy as np
from collections import Counter



#how to generate a demo from a start location?
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




#how to generate feature counts for a demonstration?
def generate_feature_counts(traj, mdp):
    """ generate feature counts from a trajectory"""
    #count each time a state was visited 
    counts = Counter({feature:0 for feature in mdp.features})
    for state,action in traj:
        counts[mdp.observe_features(state)] += 1
    
    return [counts[feature] for feature in mdp.features]



#how to generate the mu vectors for a given (s,a) in D
def generate_half_space_normals(traj, policy, mdp):
    #TODO I could probably generically write this to find a dictionary of all action: counts
    """given a trajectory, assume first element is (s,a)
    and calculate half spaces mu_pi,s_a - mu_pi,s_b for all b
    return array of np.arrays one for each halfspace normal vector
    """
    init_state,init_action = traj[0]
    #print('init_state',init_state, 'init_action',init_action)
    #calculate \bar{\mu}_\pi,s_a
    f_counts = generate_feature_counts(traj,mdp)
    #get feature counts in the order specified by mdp.features
    mu_sa = np.array(f_counts)
    #print('mu_sa',mu_sa)

    #get mu_sb for all other actions starting at state s and following policy
    actions = list(mdp.actions(init_state))
    #print(actions)
    actions.remove(init_action)
    
    mu_normals = []
    for a in actions:
        #print('a', a)
        new_start = mdp.go(init_state, a)
        #print('new',new_start)
        demo_b = generate_demonstration(new_start, policy, mdp)
        demo_b = [(init_state, a)] + demo_b
        #print(demo_b)
        mu_sb = np.array(generate_feature_counts(demo_b, mdp))
        #print('mu_sb',mu_sb)
        mu_normals.append(mu_sa - mu_sb)

    return mu_normals    
    
#generate_half_space_normals for each (s,a) in D
def generate_all_constraints(traj,policy,mdp):
    """given a demonstration, generate all the halfspace constraints along entire 
    trajectory
    """
    #print('generating all constraints')
    constraints = []
    traj_tmp = list(traj)
    #print(traj_tmp)
    #compute halfspace normals for all (s,a) pairs until terminal
    while(len(traj_tmp)>1):
        constraints += generate_half_space_normals(traj_tmp,policy,mdp)
        #print(constraints)
        traj_tmp.pop(0)
        #print('after pop',traj_tmp)
    return constraints
