import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
import utils


# computing BSC
def bsc(x, p):
    # init w as a copy of the input
    w = x.copy()
    # every index of inpuit array x is flipped with prob p
    flip_index = (np.random.random(len(w)) <= p)
    # flipping the indexes marked before
    w[flip_index] = 1 ^ w[flip_index]
    # the returned value is a np.array
    return w