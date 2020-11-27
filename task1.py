import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
from utils import perms, bin_array_to_str


# computing y and z
def uniform_channel(x):

    possible_perms = perms() #pretty fast algorithm to calculate all different perms for hamming dist<=3, without repetitions
    ind_b = np.random.randint(x.shape[0] + 1) #random from 0 to 7, if 7 the noise is the all-0 vector
    noise_b = np.zeros(x.shape, dtype=int)
    if(ind_b != len(x)):
        noise_b[ind_b] = 1
    ind_e=np.random.randint(len(possible_perms)) #draw a random perm with uniform prob.

    # those are the values of y and z computed by the uniform channel 
    y = np.bitwise_xor(x, noise_b)
    z = np.bitwise_xor(x, possible_perms[ind_e])

    # returned values are np.arrays
    return y, z


# computing p_x|z and its statistics
def pmd_z(x, iter):

    # lz contains every #iter realization of z from a input x
    lz = np.array([], dtype=int)
    # filling of lz with z realization converted to strings
    while lz.shape[0] < iter:
        lz = np.append(lz, bin_array_to_str(uniform_channel(x)[1]))

    # unique is a np.array which contains all the unique realization -> its a nparray containing strings
    # count is a np.array which stores for each unique value the number of times it was present in lz
    unique, count = np.unique(lz, return_counts=True)
    # pmd destribution of each unique z realization over the number of realizations -> count is the norm factor
    pmd = count/np.sum(count)

    # returned values are np.arrays
    return unique, pmd