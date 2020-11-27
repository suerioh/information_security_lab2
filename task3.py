import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
from utils import str_to_bin_array, bit_complement


codewords = ['0000000', '1000110', '0100101', '0010011', '0001111', '1100011', '1010101', '1001001',
    '0110110', '0101010', '0011100', '1110000', '1101100', '1011010', '0111001', '1111111']

# decodes x^ in the most probable d^
def dec(x):

    temp = ''
    min = np.inf
    for element in codewords:
        # for each codeword x, compute the hamming distance from 0 to the result of x XOR x^
        # store into temp the x which corresponds to the lowest d 
        # the *7 is necessary in order to get the real distance value (otherwise its just a rate of dissimilar bits)
        d = distance.hamming(np.bitwise_xor(str_to_bin_array(x), str_to_bin_array(element)), np.zeros(len(element)))*7
        if d < min:
            temp = element
            min = d
    # return string of the corresponding d, following task 3 conditions 
    if temp[:1] == '0':
        return temp[1:4]
    else:
        return bit_complement(temp[1:4])
    # a string is returned