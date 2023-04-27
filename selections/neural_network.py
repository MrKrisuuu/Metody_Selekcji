from jmetal.operator.selection import *  # Selekcja

from typing import List, TypeVar
from .utils import save

import tensorflow as tf
from jmetal.config import store
from itertools import combinations

import numpy as np


S = TypeVar("S")


class MyNeuralNetworkSelection(Selection[List[S], S]):
    def __init__(self):
        super(MyNeuralNetworkSelection, self).__init__()
        self.flag_list = True

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
            for x, y in combinations(range(len(front)), 2):
                individuals.append(front[x].variables + front[y].variables)
            sols = self.model.predict(individuals)
            for i, (x, y) in enumerate(combinations(range(len(front)), 2)):
                val = sols[i][0]
                matrix[x][y] = val
                matrix[y][x] = val
            sols = [sol.objectives[0] for sol in front]
            sums = [sum(row) for row in matrix]
            # save(sols, sums, number, self.get_name())
            my_result = list(zip(front, sums))
            my_result.sort(key=lambda ind: ind[1], reverse=True)
            result = [sol for sol, _ in my_result]
        return result[:number]

    def get_name(self) -> str:
        return "MyNeuralNetworkSelection"

    def train_model(self, problem):
        training_population = 150
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    2 * problem.number_of_variables, activation="relu"
                ),
                tf.keras.layers.Dense(
                    2 * problem.number_of_variables, activation="relu"
                ),
                tf.keras.layers.Dense(problem.number_of_variables, activation="relu"),
                tf.keras.layers.Dense(problem.number_of_variables, activation="relu"),
                tf.keras.layers.Dense(1, activation=None),
            ]
        )
        model.compile(optimizer="adam", loss="mean_squared_error")

        population = [
            store.default_generator.new(problem) for _ in range(training_population)
        ]
        x_train = []
        y_train = []

        for _ in range(training_population**2):
            a, b = np.random.choice(population, 2)
            result = (
                problem.evaluate(b).objectives[0] - problem.evaluate(a).objectives[0]
            )
            x_train.append(a.variables + b.variables)
            y_train.append([result])
        model.fit(x_train, y_train, epochs=10)
        self.model = model
