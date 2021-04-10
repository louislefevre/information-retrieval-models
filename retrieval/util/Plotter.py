from collections import Counter

from matplotlib import pyplot as plt
from tabulate import tabulate

from util.FileManager import write_txt


def zipfs(counter: Counter):
    prob_distribution = []
    rows = []
    total_count = sum(counter.values())
    c = 0.0

    for rank, (word, freq) in enumerate(counter.most_common(100)):
        rank += 1
        p = freq / total_count
        pr = rank * p
        c += pr
        prob_distribution.append(p)
        rows.append([word, freq, rank, "{:.3f}".format(p), "{:.3f}".format(pr)])

    _plot_distribution(prob_distribution)
    _report_parameters(rows, c)


def _plot_distribution(prob_distribution: list[float]):
    zipf_distribution = [0.1 / i for i in range(1, 101)]
    _generate_figure(prob_distribution, zipf_distribution, title="Zipf's Law", x_label="Rank",
                     y_label="Probability", file_name='zipf-plot.png')


def _report_parameters(rows: list[list], c: float):
    table = tabulate(rows, headers=['Word', 'Freq', 'r', 'Pr', 'r*Pr'])
    c = round(c / len(rows), 3)
    data = table + f'\n\nc = {c}'
    write_txt('zipf-parameters.txt', data)


def _generate_figure(*data, title=None, x_label=None, y_label=None, file_name='figure.png'):
    for d in data:
        plt.plot(d)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(file_name)
