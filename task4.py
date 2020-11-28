import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
from utils import str_to_bin_array, bin_array_to_str
from task1 import uniform_channel
from task2 import enc


# ----------- TASK 4 - EMPIRICAL CALCULATIONS OF p(z|u) , p(z) , p(z,u) , I(z,u) ------------
def task4_results():
    
    distribution_u=[] #this will contain one conditional distribution p(z|u) for every message u ∈ M
    joint_distribution=np.zeros((2**3,2**7)) #this will contain the joint distribution of z and u -> p(z,u)
    mutual_information=0

    iter=10000 #number of iterations of the conditional distribution gathering process
    for i in range(2**3): #for every message u
        u= bin_array_to_str(bin(i)[2:].zfill(3))
        distribution_z = np.zeros(2**7) #this will contain the temporary p(z|u) that will be later saved in distribution_u
        for j in range(iter): #gathering the distribution p(z|u) for the specific u
            x=str_to_bin_array(enc(u))
            z=uniform_channel(x)[1]
            distribution_z[int(bin_array_to_str(z),2)]+=1
        distribution_z= distribution_z/iter
        distribution_u.append(distribution_z) #append the distribution p(z|u) in the right position (the index is the integer conversion of u)
        plt.figure(i+1)
        plt.title("Conditional Distribution p(z|u) for u: "+ u)
        plt.ylabel("Probability")
        plt.xlabel("Values of z")
        plt.ylim(0,0.2) #capped at lim_p= 0.2, to make the values look distinguishable from the x axis
        plt.plot(range(distribution_z.shape[0]),distribution_z,'.') #one plot for every distribution
    total_distribution_z= np.zeros(2**7) #this will contain the distribution p(z)
    for i in range(len(total_distribution_z)): #calculation of p(z) (need to check every z in Z)
        for d_z in distribution_u:  #computed as sum of the conditional probabilities p(z|u) multiplied
                                    # by the probability to have that specific u,
                                    # (1/8 in our case, assuming uniformity)
                                    #for every u
            total_distribution_z[i]+= (1/8)*d_z[i] #1/8 is the probability to have the specific message u (assumed uniform)
    plt.figure(9) #plot of the distribution p(z) , we expect uniformity
    plt.title("Distribution p(z)")
    plt.ylabel("Probability")
    plt.xlabel("Values of z")
    plt.plot(range(total_distribution_z.shape[0]),total_distribution_z,'.')
    plt.ylim(0, 0.2) #capped at lim_p= 0.2, to make the values look distinguishable from the x axis
    #calculation of joint distribution and mutual information
    #for each z ∈ Z , u ∈ M
    for i in range(2**3):
        for j in range(2**7):
            joint_distribution[i][j] = distribution_u[i][j]*(1/8) #using the formula p(z,u) = p(z|u)*p(u)
            #we use the joint distribution that we have just computed, to calculate the mutual information term related to it
            mutual_information+= joint_distribution[i][j] * math.log2(joint_distribution[i][j] / (total_distribution_z[j]*(1/8)))
    plt.show() #plot all the figures
    return ('----------- TASK 4 -----------\n\nMutual Information I(u,z)= '+str(mutual_information))
