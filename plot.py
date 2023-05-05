import matplotlib.pyplot as plt

from statistics import mean, stdev


def plot_iterations(problems, selections, steps, times, path="./results"):
    for problem in problems:
        for selection in selections:
            res = [[] for _ in range(steps + 1)]
            total_time = 0
            with open(f"{path}/{problem.get_name()}/{selection.get_name()}.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    result = line.split(';')
                    result = [float(res) for res in result]
                    for i, val in enumerate(result[:-1]):
                        res[i].append(val)
                    total_time += result[-1] / times
                avg = [mean(epoch) for epoch in res]
            print(f"Total time for {selection.get_name()}: {total_time} seconds in {problem.get_name()}")
            plt.plot(range(steps + 1), avg, label=selection.get_name())
        plt.title(f"{problem.get_name()}")
        plt.yscale("log")
        plt.xlabel("Number of iterations")
        plt.ylabel("Found solution")
        plt.legend()
        plt.show()


def plot_single(problems, selection, steps, times):
    for problem in problems:
        with open(f"{problem.get_name()}/{selection.get_name()}.txt", "r") as f:
            res = [[] for _ in range(steps + 1)]
            total_time = 0
            lines = f.readlines()
            for line in lines:
                result = line.split(';')
                result = [float(res) for res in result]
                for i, val in enumerate(result[:-1]):
                    res[i].append(val)
                total_time += result[-1] / times
            avg_max = [max(epoch) for epoch in res]
            avg_sigma_plus = [mean(epoch) + stdev(epoch) for epoch in res]
            avg = [mean(epoch) for epoch in res]
            avg_sigma_minus = [mean(epoch) - stdev(epoch) for epoch in res]
            avg_min = [min(epoch) for epoch in res]
        print(f"Total time for {selection.get_name()}: {total_time} seconds in {problem.get_name()}")
        plt.plot(range(steps + 1), avg_max, linewidth=0.5, color="black")
        plt.plot(range(steps + 1), avg_sigma_plus, linewidth=0.5, color="red")
        plt.plot(range(steps + 1), avg, label=selection.get_name(), color="blue")
        plt.plot(range(steps + 1), avg_sigma_minus, linewidth=0.5, color="red")
        plt.plot(range(steps + 1), avg_min, linewidth=0.5, color="black")
        plt.title(f"{problem.get_name()}")
        plt.yscale("log")
        plt.xlabel("Number of iterations")
        plt.ylabel("Found solution")
        plt.legend()
        plt.show()


def plot_stdev(problems, selections, steps, path="./stdevs"):
    for problem in problems:
        for selection in selections:
            res = [[] for _ in range(steps + 1)]
            with open(f"{path}/{problem.get_name()}/{selection.get_name()}.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    result = line.split(';')
                    result = [float(res) for res in result[:-1]]
                    for i, val in enumerate(result):
                        res[i].append(val)
                stdevs = [mean(epoch) for epoch in res]
            plt.plot(range(steps + 1), stdevs, label=selection.get_name())
        plt.title(f"{problem.get_name()}")
        plt.xlabel("Number of iterations")
        plt.ylabel("Stdev of solutions")
        plt.legend()
        plt.show()

