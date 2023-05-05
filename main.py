from typing import Callable, List
import os
import shutil

from MyGeneticAlgorithm import MyGeneticAlgorithm
from MyFunctions import Sphere, Rastrigin, Rosenbrock, Schwefel, Griewank
from plot import plot_iterations, plot_single, plot_stdev
from selections import (
    MyNormalPairwiseComparisonSelection,
    MyCauchyPairwiseComparisonSelection,
    MyNormalSelection,
    MyCauchySelection,
    MyNormalFadingSelection,
    MyCauchyFadingSelection,
    MyBestSolutionSelection,
    BinaryTournamentSelection,
    RandomSolutionSelection,
)
from jmetal.core.problem import Problem
from jmetal.core.operator import Selection


from jmetal.config import store


def create_initial_solutions(problem, population_size):
    return [store.default_generator.new(problem) for _ in range(population_size)]


def prepare_files(path, selections):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for selection in selections:
        open(f"{path}/{selection.get_name()}.txt", "x")


def run_selections(problems: List[Problem],
                   selections: List[Selection],
                   times: int,
                   population_size: int,
                   offspring_population_size: int,
                   steps: int,
                   path= "./results"):
    for problem in problems:
        prepare_files(f"{path}/{problem.get_name()}", selections)
        prepare_files(f"./stdevs/{problem.get_name()}", selections)
    for problem in problems:
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

                with open(f"{path}/{problem.get_name()}/{selection.get_name()}.txt", "a") as f:
                    for solution in solutions:
                        f.write(str(solution.objectives[0]))
                        f.write(';')
                    f.write(f"{algorithm.total_computing_time}")
                    f.write('\n')

                print(
                    f"{i+1}. {selection.get_name()} for {problem.get_name()}: {best_solution.objectives[0]} in {algorithm.total_computing_time} seconds"
                )


if __name__ == "__main__":
    problems = [
        Sphere(100),
        Rastrigin(100),
        Rosenbrock(100),
        Schwefel(100),
        Griewank(100)
    ]
    selections = [
        MyNormalPairwiseComparisonSelection(2),
        MyCauchyPairwiseComparisonSelection(0.2),
        MyNormalSelection(0.25),
        MyCauchySelection(0.025),
        MyNormalFadingSelection(0.5, 0.999),
        MyCauchyFadingSelection(0.05, 0.999),
        MyBestSolutionSelection(),
        BinaryTournamentSelection(),
        RandomSolutionSelection(),
    ]
    times = 10
    population_size = 100
    offspring_population_size = 30
    steps = 2000
    # run_selections(problems, selections, times, population_size, offspring_population_size, steps)
    plot_iterations(problems, selections, steps, times)
    plot_stdev(problems, selections, steps)
    #plot_single(problems, MyCauchySelection(), steps, times)


# TODO
# Reguła 5 sukcesów -> zrobić
# niepełne porównania -> przetestować (niekompletne macierze porównywania parami)