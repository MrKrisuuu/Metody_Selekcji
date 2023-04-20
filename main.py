from typing import Callable, List

from MyGeneticAlgorithm import MyGeneticAlgorithm
from MyFunctions import RastriginFunction, Sphere
from selections import (
    MyCauchyFadingSelection,
    MyCauchySelection,
    MyNeuralNetworkSelection,
    MyNormalPairwiseComparisonSelection,
    MyOptimizedNormalPairwiseComparisonSelection,
    MyNormalSelection,
    MyNormalFadingSelection,
    BestSolutionSelection,
    BinaryTournamentSelection,
    RandomSolutionSelection,
)
from jmetal.core.problem import Problem
from jmetal.core.operator import Selection


from jmetal.config import store

import matplotlib.pyplot as plt

from statistics import mean


def create_initial_solutions(problem, population_size):
    return [store.default_generator.new(problem) for _ in range(population_size)]


def run_selections(problems: List[Problem], selections: List[Selection], times: int):
    population_size = 100
    offspring_population_size = 30
    steps = 500
    for problem in problems:
        keys = [selection.get_name() for selection in selections]

        results = {
            key: {
                "times": [],
                "solutions": [],
            }
            for key in keys
        }

        for i in range(times):
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

                current_results = results[selection.get_name()]

                current_results["solutions"].append(
                    [solution.objectives[0] for solution in solutions]
                )
                current_results["times"].append(algorithm.total_computing_time)
                print(
                    f"{i+1}. {selection.get_name()} for {problem.get_name()}: {best_solution.objectives[0]} in {algorithm.total_computing_time} seconds"
                )
        for selection_name in results:
            avg = [0 for _ in range(steps + 1)]
            for res in results[selection_name]["solutions"]:
                for i, val in enumerate(res):
                    avg[i] += val / times
            total_time = mean(results[selection_name]["times"])
            print(f"Total time for {selection_name}: {total_time} seconds")
            plt.plot(range(steps + 1), avg, label=selection_name)
        plt.yscale("log")
        plt.legend()
        plt.savefig(f"./result.png")
        plt.show()


if __name__ == "__main__":
    problems = [RastriginFunction(100), Sphere(100)]
    selections = [
        # MyNeuralNetworkSelection(),
        MyNormalPairwiseComparisonSelection(),
        MyOptimizedNormalPairwiseComparisonSelection(),
        MyNormalSelection(),
        MyNormalFadingSelection(),
        MyCauchySelection(),
        MyCauchyFadingSelection(),
        BestSolutionSelection(),
        BinaryTournamentSelection(),
        RandomSolutionSelection(),
    ]
    times = 3
    run_selections(problems, selections, times)


# skupić się na różnych rozkłądach (kosziego)
# Estimation of dis(?) EDA
# Reguła 5 sukcesów
