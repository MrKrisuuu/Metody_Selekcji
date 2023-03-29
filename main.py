from MyGeneticAlgorithm import MyGeneticAlgorithm
from MyFunctions import *
from MySelections import *

from jmetal.config import store

import matplotlib.pyplot as plt


def create_initial_solutions(problem, population_size):
    return [store.default_generator.new(problem)
            for _ in range(population_size)]


def run_selections(problems, selections):
    population_size = 100
    offspring_population_size = 100
    steps = 100
    for problem in problems:
        initial_solutions = create_initial_solutions(problem, population_size)
        for selection in selections:
            algorithm = MyGeneticAlgorithm(
                steps=steps,
                problem=problem,
                population_size=population_size,
                offspring_population_size=offspring_population_size,
                selection=selection
            )
            solutions = algorithm.run(initial_solutions)
            best_solution = algorithm.get_result()
            solutions.append(best_solution)
            # for i, solution in enumerate(solutions):
            #     print(f"{i}:", solution)
            results = [solution.objectives[0] for solution in solutions]
            print(f"{selection.get_name()} for {problem.get_name()}: {best_solution.objectives[0]} in {algorithm.total_computing_time} seconds")
            plt.plot(range(steps + 1), results, label=selection.get_name())
        plt.yscale("log")
        plt.legend()
        plt.show()


problems = [RastriginFunction(3)]
selections = [BestSolutionSelection(),
              BinaryTournamentSelection(),
              RandomSolutionSelection()]
run_selections(problems, selections)
