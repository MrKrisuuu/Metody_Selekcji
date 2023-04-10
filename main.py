from typing import Callable, List

import tensorflow as tf
from MyGeneticAlgorithm import MyGeneticAlgorithm
from MyFunctions import RastriginFunction, Sphere
from MySelections import (
    MyNeuralNetworkSelection,
    MyRandomSelection,
    BestSolutionSelection,
    BinaryTournamentSelection,
    RandomSolutionSelection,
)
from jmetal.core.problem import Problem
from jmetal.core.operator import Selection


from jmetal.config import store

import matplotlib.pyplot as plt


def create_initial_solutions(problem, population_size):
    return [store.default_generator.new(problem) for _ in range(population_size)]


def run_selections(
    problems: List[Problem],
    selections: List[Selection],
):
    population_size = 50
    offspring_population_size = 20
    steps = 100
    for problem in problems:
        initial_solutions = create_initial_solutions(problem, population_size)
        for selection in selections:
            algorithm = MyGeneticAlgorithm(
                steps=steps,
                problem=problem,
                population_size=population_size,
                offspring_population_size=offspring_population_size,
                selection=selection,
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


if __name__ == "__main__":
    problems = [
        RastriginFunction(100),
        #Sphere(10)
    ]
    selections = [
        MyNeuralNetworkSelection(),
        MyRandomSelection(),
        BestSolutionSelection(),
        BinaryTournamentSelection(),
        RandomSolutionSelection(),
    ]

    run_selections(problems, selections)
