import mdp


mdp1 = mdp.GridMDP([[+10, -1],
                    [-5, -1]],
                      terminals=[(0,1)])
                      
mdp2 = mdp.GridMDP([[+10, -3],
                    [-2, -1]],
                      terminals=[(0,1)])
pi1, U1 = mdp.policy_iteration(mdp1)
print "pi1"
print pi1
print "U1"
print U1
pi2, U2 = mdp.policy_iteration(mdp2)
print "pi2"
print pi2
print "U2"
print U2

#why is this true?
new_u = mdp.policy_evaluation(pi1, U1, mdp2, 100)
print "new_u", new_u
pi_new = mdp.best_policy(mdp2, new_u)
print "pi_new"
print pi_new


                      
                      
