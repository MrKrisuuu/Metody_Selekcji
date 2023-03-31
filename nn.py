import tensorflow as tf

import numpy as np

from jmetal.config import store


def build(problem):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(2 * problem.number_of_variables, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation=None),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def train(problem, population_size, output_size):
    population = [store.default_generator.new(problem) for _ in range(population_size)]

    x_train = []
    y_train = []
    for _ in range(output_size):
        a, b = np.random.choice(population, 2)
        result = problem.evaluate(a).objectives[0] - problem.evaluate(b).objectives[0]
        x_train.append(a.variables + b.variables)
        y_train.append([result])
    print(np.array(x_train).shape)
    return x_train, y_train
