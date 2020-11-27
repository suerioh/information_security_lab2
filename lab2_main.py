import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
from utils import str_to_bin_array
from task1 import pmd_z, uniform_channel, bin_array_to_str
from task2 import enc
from task3 import dec
from task5 import bsc
from task6 import ideal_sec_capacity


def main():

    # ----------- TASK 1 - UNIFORM CHANNEL -----------
    # plot pmd_z|x
    print('----------- TASK 1 -----------')
    print('stats pmd_z using x = 1001000')

    pmdz = pmd_z(str_to_bin_array('1001000'), 10000)[1]

    # unique realizations is the len of pmd
    print('unique z realizations =', pmdz.shape[0])
    # computed the pmd mean and variace
    print('mean value =', np.mean(pmdz))
    print('variance value =', np.var(pmdz))
    # print(unique_values_z) <- debugging purposes

    # plotting all data acquired
    mean_arr = np.full((pmdz.shape[0],), np.mean(pmdz))
    # n contains all integers from o to length of pmd, it is used to plot numbers instead of z values
    n = np.arange(0, pmdz.shape[0], step=1)

    plt.plot(n, pmdz, '.', label='original data')
    plt.plot(n, mean_arr, 'r', label='mean value')
    plt.ylim(0, np.mean(pmdz)*3)
    plt.legend()
    plt.title('pmd_z|x')
    plt.xlabel('z values')
    plt.ylabel('prob')
    plt.show()




    # ----------- TASK 2 - ENCODER -----------
    codewords = ['0000000', '1000110', '0100101', '0010011', '0001111', '1100011', '1010101', '1001001',
    '0110110', '0101010', '0011100', '1110000', '1101100', '1011010', '0111001', '1111111']

    # conversion of codewords in np.arrays
    c = np.array([])
    for element in codewords:
        c = np.append(c, str_to_bin_array(element))


    # try encoder with random messages
    print('\n----------- TASK 2 -----------')
    print('verify encorder using random input d')
    # compute a random message d
    msg = np.random.randint(0, 2, 3, dtype=int)
    msg = bin_array_to_str(msg)
    print('d =', msg) 
    print('x =', enc(msg))




    # ----------- TASK 3 - DECODER -----------
    # try decoder
    print('\n----------- TASK 3 -----------')
    print('verify cascading encoder + decoder with random input d')
    msg = np.random.randint(0, 2, 3, dtype=int)
    msg = bin_array_to_str(msg)
    print('d =', msg)
    print('x =', enc(msg))
    print('d^ =', dec(enc(msg)))

    print('\nverify cascading encoder + legitimate channel + decoder with random input d')
    msg = np.random.randint(0, 2, 3, dtype=int)
    msg = bin_array_to_str(msg)
    print('d =', msg)
    print('x =', enc(msg))

    y = uniform_channel(str_to_bin_array(enc(msg)))[0]

    print('y =', bin_array_to_str(y))
    print('d^ =', dec(bin_array_to_str(y)))




    # ----------- TASK 5 - BSC CHANNEL -----------
    # try BSC channel, using a very long binary sequence and constant ε, δ values 
    epsilon = 0.1
    delta = 0.4

    # sequence of 10000 bits
    x = np.random.randint(0, 2, 100000)
    y = bsc(x, epsilon)
    z = bsc(x, delta)

    print('\n----------- TASK 5 -----------')
    print('verify implementation of BSC channel')
    print('ε = ', epsilon)
    print('δ = ', delta)

    # scipy.spatial.distance.hamming gives already the error rate (dissimilar bit rate) from real sequence to y and z
    print('bit error rate in a sequence of 100000 bits, with error probability ε =', round(distance.hamming(x, y), 4))
    print('bit error rate in a sequence of 100000 bits, with error probability δ =', round(distance.hamming(x, z), 4))


    # try encoding/decoding using BSC channel
    print('\nverify cascading encoder + BSC channel + decoder with random input d')
    msg = np.random.randint(0, 2, 3, dtype=int)
    epsilon = 0.3
    msg = bin_array_to_str(msg)

    print('ε = ', epsilon)
    print('d =', msg)
    print('x =', enc(msg))

    y = bsc(str_to_bin_array(enc(msg)), epsilon)

    print('y =', bin_array_to_str(y))
    print('d^ =', dec(bin_array_to_str(y)))




    # ----------- TASK 6 - BSC EVALUATION -----------
    # evaluation the resulting reliability in BCS transmissions
    print('\n----------- TASK 6 -----------')

    epsilon = 0
    count = 0
    error = np.array([])

    # for each value of epsilon from 0 to 1 step 0.01, computing average P[u≠û]
    # P[u≠û] for a specific epsilon is computed over 100 random messages d
    while epsilon <= 1:
        for i in range (100):
            msg = bin_array_to_str(np.random.randint(0, 2, 3, dtype=int))
            y = bsc(str_to_bin_array(enc(msg)), epsilon)
            # if d and d^ dont match, count is increased 
            if msg != dec(bin_array_to_str(y)):
                count += 1
        
        # error contains the average P[u≠û] for each epsilon
        error = np.append(error, count/100)
        # print('precision with ε =', round(epsilon, 3), 'is: ', round(1-(count/100), 4)) <- debugging purposes
        
        # reinit count and increment epsilon
        count = 0
        epsilon += 0.01


    # plot error decoding probability
    n = np.arange(0, 1, step=0.01)
    n_label = np.arange(0, 1.01, step=0.1)

    plt.plot(n, error, '.')
    plt.ylim(0, 1)
    plt.title('error decoding probability')
    plt.xlabel('ε')
    plt.xticks(n_label)
    plt.ylabel('P[u≠û]')
    plt.show()


    '''
    # mutual information as a function of delta
    # i work from 0.01 to 0.99 to avoid log problems with prob equal to 0 or 1
    delta = 0.01
    mut_information = np.array([])

    while delta <= 0.99:
        mut_information = np.append(mut_information, mutual(delta))
        delta += 0.01
    '''


    # hardcoded outputs, in order not to spend 10min time every run
    mut_information_epsilon = np.array([2.98727175, 2.94614793, 2.89240417, 2.82188483, 
    2.7279838, 2.63924372, 2.52294672, 2.41640956, 2.3035355, 2.19209787, 
    2.06848962, 1.94878108, 1.82283085, 1.70219992, 1.57787233, 1.45875622, 
    1.34523906, 1.23829645, 1.13559747, 1.02173213, 0.92142953, 0.83188522, 
    0.74241697, 0.66807166, 0.58488147, 0.51244587, 0.4476626,  0.38464666, 
    0.33932902, 0.28316366, 0.23466681, 0.19716323, 0.16514119, 0.12990702, 
    0.10772678, 0.08557285, 0.0641337,  0.05269767, 0.03978973, 0.03084968, 
    0.02205296, 0.01806612, 0.01431043, 0.0109065, 0.00967108, 0.00880088, 
    0.00826041, 0.00821538, 0.00839826, 0.00874734, 0.00748919, 0.00815293, 
    0.00858326, 0.00855921, 0.009666, 0.01166269, 0.0131119, 0.01777343, 
    0.02270988, 0.02985923, 0.0407703, 0.04957619, 0.06559732, 0.08401337, 
    0.10360239, 0.13391682, 0.16532999, 0.19567226, 0.23536821, 0.27651931, 
    0.32466227, 0.38237019, 0.44553695, 0.50725782, 0.58774841, 0.66218796, 
    0.75173272, 0.83202576, 0.9218408, 1.0249672, 1.12588786, 1.24243351, 
    1.3604682, 1.4503219, 1.58194663, 1.70088592, 1.82696584, 1.94984347, 
    2.06786747, 2.19720531, 2.30412037, 2.41973887, 2.53932641, 2.62947277, 
    2.73380045, 2.82580941, 2.89139009, 2.94767537])


    mut_information_delta = np.array([2.98645705, 2.94715407, 2.89093467, 2.82197489, 2.73132822, 2.64164925,
    2.5309403,  2.42038138, 2.31436616, 2.18165529, 2.07059369, 1.95582943,
    1.82122496, 1.70241707, 1.58122068, 1.46668769, 1.35004905, 1.22964376,
    1.13723254, 1.02195306, 0.92396771, 0.83526219, 0.74651466, 0.66087588,
    0.58038733, 0.51295607, 0.43996717, 0.38199166, 0.33071199, 0.27693776,
    0.23596125, 0.19945505, 0.16111749, 0.13383371, 0.10527193, 0.08356272,
    0.06502537, 0.05239463, 0.0386195,  0.03073973, 0.02296977, 0.01715279,
    0.01373081, 0.0098756,  0.00914994, 0.00802845, 0.0083037,  0.0077014,
    0.00841362, 0.00829982, 0.0089077,  0.00831979, 0.00747178, 0.00914853,
    0.0091277,  0.01103066, 0.0134162,  0.01736062, 0.02209025, 0.0313768,
    0.03786967, 0.05029615, 0.06599428, 0.08425918, 0.10456138, 0.12867665,
    0.16103165, 0.19863727, 0.24157585, 0.27593007, 0.32661139, 0.38769061,
    0.44267463, 0.50368331, 0.58201888, 0.66214209, 0.74000942, 0.82432058,
    0.93019579, 1.02468489, 1.11659644, 1.23040818, 1.34097607, 1.46163171,
    1.57905414, 1.7037856,  1.82151823, 1.95291896, 2.05545181, 2.19340341,
    2.31774142, 2.42681032, 2.53152046, 2.6258033,  2.72359376, 2.81556758,
    2.89034845, 2.94439152])


    # plot mutual information as a function of delta
    n = np.arange(0.01, 0.99, step=0.01)
    n_label = np.arange(0, 1.01, step=0.1)

    plt.plot(n, mut_information_delta, '.')
    plt.ylim(0, 3)
    plt.title('mutual information as a function of delta')
    plt.xlabel('δ')
    plt.xticks(n_label)
    plt.ylabel('I(u;z)')
    plt.show()


    # compute ideal secrecy capacity for each delta and epsilon pair
    e = 0.01
    d = 0.01
    ideal_sec_capacity_arr = np.zeros((98, 98))

    # computing ideal capacity matrix based on different values of epsilon and delta
    for i in range(98):
        d = 0.01
        for j in range(98):
            ideal_sec_capacity_arr[i, j] = ideal_sec_capacity(e, d)
            d += 0.01
        e += 0.01


    # plot ideal secrecy capacity for each delta and epsilon pair
    eps = np.arange(0.01, 0.99, step=0.01)
    delts = np.arange(0.01, 0.99, step=0.01)
    levels = np.arange(0, 1.1, step=0.1)
    plt.contour(eps, delts, ideal_sec_capacity_arr, levels, linewidths=1, cmap='RdYlGn')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    n_label = np.arange(0, 1.1, step=0.1)
    plt.xticks(n_label)
    plt.yticks(n_label)
    plt.title('ideal secrecy capacity')
    plt.xlabel('ε')
    plt.ylabel('δ')
    plt.colorbar()
    plt.show()


    # compute real secrecy capacity
    real_sec_capacity_arr = np.zeros((98, 98))

    # computing real capacity matrix based on different values of epsilon and delta
    for i in range(98):
        for j in range(98):
            real_sec_capacity_arr[i, j] = mut_information_delta[i] - mut_information_epsilon[j]


    # plot real secrect capacity
    eps = np.arange(0.01, 0.99, step=0.01)
    delts = np.arange(0.01, 0.99, step=0.01)
    levels = np.arange(0, 1.1, step=0.1)
    plt.contour(eps, delts, np.transpose(real_sec_capacity_arr), levels, linewidths=1, cmap='RdYlGn')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    n_label = np.arange(0, 1.1, step=0.1)
    plt.xticks(n_label)
    plt.yticks(n_label)
    plt.title('real secrecy capacity')
    plt.xlabel('ε')
    plt.ylabel('δ')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()