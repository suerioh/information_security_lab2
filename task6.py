import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
from utils import bin_array_to_str, str_to_bin_array
from task5 import bsc
from task2 import enc


# mutual information for the BSC chennel
def mutual(p):    
    distribution_u = [] #this should contain one distribution for every message u
    joint_distribution = np.zeros((2**3,2**7)) #this should contain the joint distribution of z and u
    mutual_information = 0
    iter = 10000
    for i in range(2**3):
        u = (bin(i)[2:].zfill(3))
        distribution_z = np.zeros(2**7) # this will contain the distribution of z for a single u
        for j in range(iter):
            x = str_to_bin_array(enc(u))
            z = bsc(x, p)
            distribution_z[int(bin_array_to_str(z),2)]+=1
        distribution_z = distribution_z/iter
        distribution_u.append(distribution_z)
    total_distribution_z= np.zeros(2**7)
    for i in range(len(total_distribution_z)):
        for d_z in distribution_u:
            total_distribution_z[i]+= (1/8)*d_z[i] #1/8 is the probability to have the specific message u (assumed uniform)
    for i in range(2**3):
        for j in range(2**7):
            joint_distribution[i][j] = distribution_u[i][j]*(1/8)
            mutual_information+= joint_distribution[i][j] * math.log2(joint_distribution[i][j] / (total_distribution_z[j]*(1/8)) + 0.000001)

    return mutual_information


# ideal secrecy capacity
# i am using the same formla present in theory slides
def ideal_sec_capacity(epsilon, delta):
    c_ab = delta * math.log(delta, 2) + (1 - delta) * math.log(1 - delta, 2)
    c_ae = epsilon * math.log(epsilon, 2) + (1 - epsilon) * math.log(1 - epsilon, 2)

    return c_ab - c_ae