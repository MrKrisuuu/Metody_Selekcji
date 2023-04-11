from jmetal.core.operator import Selection
from jmetal.operator.selection import *  # Selekcja

from typing import List, TypeVar

import tensorflow as tf
from jmetal.config import store
from itertools import combinations

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from copy import deepcopy


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
    min_val = sums2[number-1]
    plt.plot([x_min, x_max], [min_val, min_val], color="black")
    plt.title(f"Epoch: {nr}")
    plt.savefig(f"./states_{name}/{nr}.png")
    plt.clf()
    nr += 1


class MyNeuralNetworkSelection(Selection[List[S], S]):
    def __init__(self):
        super(MyNeuralNetworkSelection, self).__init__()

    def execute(self, front: List[S], number=None) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        if len(front) == 1:
            result = front[0]
        else:
            matrix = [[0 for _ in range(len(front))] for _ in range(len(front))]
            individuals = []
            for (x, y) in combinations(range(len(front)), 2):
                individuals.append(front[x].variables + front[y].variables)
            sols = self.model.predict(individuals)
            for (i, (x, y)) in enumerate(combinations(range(len(front)), 2)):
                matrix[x][y] = sols[i][0]
                matrix[y][x] = -sols[i][0]
            sols = [sol.objectives[0] for sol in front]
            sums = [sum(row) for row in matrix]
            #save(sols, sums, number, self.get_name())
            my_result = list(zip(front, sums))
            my_result.sort(key=lambda ind: ind[1], reverse=True)
            result = [sol for sol, _ in my_result]
        return result[:number]

    def get_name(self) -> str:
        return "MyNeuralNetworkSelection"

    def train_model(self, problem):
        training_population = 100
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(2 * problem.number_of_variables, activation="relu"),
                tf.keras.layers.Dense(2 * problem.number_of_variables, activation="relu"),
                tf.keras.layers.Dense(problem.number_of_variables, activation="relu"),
                tf.keras.layers.Dense(problem.number_of_variables, activation="relu"),
                tf.keras.layers.Dense(1, activation=None),
            ]
        )
        model.compile(optimizer="adam", loss="mean_squared_error")

        population = [store.default_generator.new(problem) for _ in range(training_population)]
        x_train = []
        y_train = []

        for _ in range(training_population**2):
            a, b = np.random.choice(population, 2)
            result = problem.evaluate(b).objectives[0] - problem.evaluate(a).objectives[0]
            x_train.append(a.variables + b.variables)
            y_train.append([result])
        model.fit(x_train, y_train, epochs=10)
        self.model = model


class MyRandomSelection(Selection[List[S], S]):
    def __init__(self):
        super(MyRandomSelection, self).__init__()

    def evaluate(self, sol1, sol2, width):
        val1 = np.random.normal(sol1.objectives[0], width)
        val2 = np.random.normal(sol2.objectives[0], width)
        return val2 - val1

    def execute(self, front: List[S], number=None) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        if len(front) == 1:
            result = front[0]
        else:
            matrix = [[0 for _ in range(len(front))] for _ in range(len(front))]
            sols = [sol.objectives[0] for sol in front]
            width = max(sols) - min(sols)
            for (x, y) in combinations(range(len(front)), 2):
                val = self.evaluate(front[x], front[y], width)
                matrix[x][y] = val
                matrix[y][x] = -val
            sums = [sum(row) for row in matrix]
            #save(sols, sums, number, self.get_name())
            my_result = list(zip(front, sums))
            my_result.sort(key=lambda ind: ind[1], reverse=True)
            result = [sol for sol, _ in my_result]
        return result[:number]

    def get_name(self) -> str:
        return "MyRandomSelection"
