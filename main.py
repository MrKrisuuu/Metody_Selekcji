from typing import Callable, List

import tensorflow as tf
from MyGeneticAlgorithm import MyGeneticAlgorithm
from MyFunctions import RastriginFunction, Sphere
from MySelections import (
    MySelection,
    MyNeuralNetworkSelection,
    BestSolutionSelection,
    BinaryTournamentSelection,
    RandomSolutionSelection,
)
from jmetal.core.problem import Problem
from jmetal.core.operator import Selection


from jmetal.config import store

import matplotlib.pyplot as plt

from nn import build, train


def create_initial_solutions(problem, population_size):
    return [store.default_generator.new(problem) for _ in range(population_size)]

    # w petli output_size
    # 2 osobnikow z populacji
    # porownac (chuj wie jak)
    # polaczyc te 2 osobniki i zapisac do x_train
    # wynik porownania zapisac do y_train
    # zwrocic [x_train, y_train]


def run_selections(
    problems: List[Problem],
    selections: List[Callable[[tf.keras.models.Sequential], Selection]],
):
    population_size = 100
    offspring_population_size = 100
    steps = 100
    for problem in problems:
        training_population = 1000
        # training_output = 50_000
        training_output = 1_000
        print(f"Compiling model for {problem.get_name()}")
        model = build(problem)
        print(f"Training model for {problem.get_name()}")
        x_train, y_train = train(problem, training_population, training_output)
        model.fit(x_train, y_train, epochs=20)

        initial_solutions = create_initial_solutions(problem, population_size)
        for selection in selections:
            algorithm = MyGeneticAlgorithm(
                steps=steps,
                problem=problem,
                population_size=population_size,
                offspring_population_size=offspring_population_size,
                selection=selection(model),
            )
            solutions = algorithm.run(initial_solutions)
            best_solution = algorithm.get_result()
            solutions.append(best_solution)
            # for i, solution in enumerate(solutions):
            #     print(f"{i}:", solution)
            results = [solution.objectives[0] for solution in solutions]
            print(
                f"{selection.get_name()} for {problem.get_name()}: {best_solution.objectives[0]} in {algorithm.total_computing_time} seconds"
            )
            plt.plot(range(steps + 1), results, label=selection.get_name())
        plt.yscale("log")
        plt.legend()
        plt.show()


problems = [
    # RastriginFunction(3),
    Sphere(5)
]
selections = [
    lambda model: MyNeuralNetworkSelection(model),
    # lambda model: MySelection(),
    # lambda model: BestSolutionSelection(),
    # lambda model: BinaryTournamentSelection(),
    # lambda model: RandomSolutionSelection(),
]

run_selections(problems, selections)
