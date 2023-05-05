import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from copy import deepcopy
import os


def save(sols, sums, number, name):
    nr = 1
    if not os.path.exists(f"./states_{name}"):
        os.makedirs(f"./states_{name}")
    while True:
        if os.path.exists(f"./states_{name}/{nr}.png"):
            nr += 1
        else:
            break
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
