import matplotlib.pyplot as plt

from statistics import mean


def plot_iterations(results, steps, times):
    for selection_name in results:
        avg = [0 for _ in range(steps + 1)]
        for res in results[selection_name]["solutions"]:
            for i, val in enumerate(res):
                avg[i] += val / times
        total_time = mean(results[selection_name]["times"])
        print(f"Total time for {selection_name}: {total_time} seconds")
        plt.plot(range(steps + 1), avg, label=selection_name)
    plt.yscale("log")
    plt.xlabel("Number of iterations")
    plt.ylabel("Found solution")
    plt.legend()
    plt.savefig(f"./result_iter.png")
    plt.show()


def plot_time(results, steps, times):
    pass
