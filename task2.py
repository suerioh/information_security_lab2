import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
from utils import bit_complement


codewords = ['0000000', '1000110', '0100101', '0010011', '0001111', '1100011', '1010101', '1001001',
    '0110110', '0101010', '0011100', '1110000', '1101100', '1011010', '0111001', '1111111']

# encoding the plaintext d
def enc(d):
    # Txu contains the two possible codewords
    Txu = []
    for element in codewords:
        # for each codeword, if the condition of task2 holds, then it is added to Txu 
        if element[:4] == ('0' + d):
            Txu.append(element)
    Txu = np.append(Txu, bit_complement(Txu[0]))

    # return one random element of Txu
    # a string is returned
    return Txu[np.random.randint(0, 2)]