import numpy as np

"""checks weights agains normal vector for halfspace"""
def check_halfspace(weights, normal):
    #print(np.dot(weights, normal))
    if np.dot(weights, normal) >= 0:
        return True
    else:
        return False

"""returns uniform sample from [-Mw,Mw]^dim space"""
def uniform_sample(Mw, dim):
    sample = 2 * Mw * np.random.rand(dim) - Mw
    #print(sample)
    return sample

"""returns true if sample is within the intersection of all constraints"""
def check_constraints(sample, constraints):
    for c in constraints:
        if not check_halfspace(sample, c):
            return False
    return True
"""returns G(D) from Cakmak paper, representing the 
   negative of uncertanity"""
def calculate_uncertainty(constraints, num_samples, Mw):
    count = 0.0
    dim = len(constraints[0])
    #print(dim)
    for s in range(num_samples):
        if check_constraints(uniform_sample(Mw, dim), constraints):
                count += 1.0
    return - count / num_samples

def find_feasible_weights(constraints, num_samples, Mw):
    feasible = []
    dim = len(constraints[0])
    for i in range(num_samples):
        while True:
            sample = uniform_sample(Mw, dim)
            if check_constraints(sample, constraints):
                feasible.append(sample)
                break
    return feasible

    

#mubar1 = np.array([-2,-1,1])
#mubar2 = np.array([2,3,1])
#mubar3 = np.array([1,-1,-3])
#mubar4 = np.array([-1,-1,1])
#mubar5 = np.array([-4,-2])
#print(check_halfspace(mubar1, mubar2))

#constraints = []
#constraints.append(mubar1)
#constraints.append(mubar2)
#constraints.append(mubar3)
#constraints.append(mubar4)
#print(constraints)
#print(constraints[0].shape)

#for x in range(10):
#    print(uniform_sample(1,4))

#print(calculate_uncertainty(constraints, 200000, 1))
