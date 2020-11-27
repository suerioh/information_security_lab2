import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math


# ----------- UTILS -----------
# converts strings like '0100101' into numpy arrays, where each entry is a different value
def str_to_bin_array(s):
	a = np.array(list(s), dtype=int)
	return a


# opposite of str_to_bin_array
def bin_array_to_str(a):
	b = ''
	for x in a:
		b += str(x)
	return b


# bit complent of a string of 0s and 1s
def bit_complement(f):
    b = ''
    for x in f:
        if x == '0':
            b += '1'
        else:
            b += '0'
    # a string is returned
    return b


# find permutations
def perms():
    perm = []
    mask = np.zeros(7,dtype='uint8')
    perm.append(mask)
    for i in range(7):
        temp=mask.copy()
        temp[i]=1
        perm.append(temp)
    for i in range(1,29):
        mask=perm[i].copy()
        startpos=0
        for j in reversed(range(7)):
            if (mask[j]==1):
                startpos=j+1
                break
        for j in range(startpos,7):
            temp=mask.copy()
            temp[j]=1
            perm.append(temp)
    perm = np.asarray(perm)
    return perm