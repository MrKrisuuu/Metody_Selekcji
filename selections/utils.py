from jmetal.operator.selection import *  # Selekcja

from typing import List, TypeVar

import tensorflow as tf
from jmetal.config import store
from itertools import combinations

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from copy import deepcopy
from scipy.stats import cauchy


S = TypeVar("S")

global nr
nr = 1


def save(sols, sums, number, name):
    global nr
    plt.scatter(sols, sums, color="red")
    reg = LinearRegression().fit([[sol] for sol in sols], sums)
    a = reg.coef_
    b = reg.intercept_
    x_min = min(sols)
    x_max = max(sols)
    plt.plot([x_min, x_max], [a * x_min + b, a * x_max + b], color="blue")
    sums2 = deepcopy(sums)
    sums2.sort(reverse=True)
    min_val = sums2[number - 1]
    plt.plot([x_min, x_max], [min_val, min_val], color="black")
    plt.title(f"Epoch: {nr}")
    plt.savefig(f"./states_{name}/{nr}.png")
    plt.clf()
    nr += 1
