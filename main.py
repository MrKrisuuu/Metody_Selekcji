from typing import Callable, List

from MyGeneticAlgorithm import MyGeneticAlgorithm
from MyFunctions import RastriginFunction, Sphere
from plot import plot_iterations, plot_time
from selections import (
    MyCauchyFadingSelection,
    MyCauchySelection,
    MyNeuralNetworkSelection,
    MyNormalPairwiseComparisonSelection,
    MyOptimizedNormalPairwiseComparisonSelection,
    MyOptimizedCauchyPairwiseComparisonSelection,
    MyNormalSelection,
    MyNormalFadingSelection,
    BestSolutionSelection,
    BinaryTournamentSelection,
    RandomSolutionSelection,
)
from jmetal.core.problem import Problem
from jmetal.core.operator import Selection


from jmetal.config import store


def create_initial_solutions(problem, population_size):
    return [store.default_generator.new(problem) for _ in range(population_size)]


def run_selections(problems: List[Problem], selections: List[Selection], times: int):
    population_size = 100
    offspring_population_size = 30
    steps = 1000
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
        plot_iterations(results, steps, times)
        # plot_time(results, steps, times)


if __name__ == "__main__":
    problems = [
        RastriginFunction(100),
        # Sphere(100)
    ]
    selections = [
        # MyNeuralNetworkSelection(),
        MyNormalPairwiseComparisonSelection(),
        MyOptimizedNormalPairwiseComparisonSelection(),
        MyOptimizedCauchyPairwiseComparisonSelection(),
        MyNormalSelection(),
        MyNormalFadingSelection(),
        MyCauchySelection(),
        MyCauchyFadingSelection(),
        BestSolutionSelection(),
        BinaryTournamentSelection(),
        RandomSolutionSelection(),
    ]
    times = 20
    run_selections(problems, selections, times)


# skupić się na różnych rozkłądach (kosziego)
# Estimation of dis(?) EDA -> zostawić
# Reguła 5 sukcesów -> zrobić


# niepełne porównania -> przetestować (niekompletne macierze porównywania parami)
# zobaczyć maila

# usunąć z dominancję
# dodać odchylenie statndardowe
# inne problemy
