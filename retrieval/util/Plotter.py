from collections import Counter

from matplotlib import pyplot as plt


def plot_frequency(counter: Counter):
    frequencies = _normalise(counter.values())
    frequencies.sort(reverse=True)
    frequencies = frequencies[:100]

    _generate_figure(frequencies, title="Term Frequency", x_label="Rank",
                     y_label="Probability", file_name='term-frequencies.png')


def _generate_figure(data, title=None, x_label=None, y_label=None, file_name='figure.png'):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_name)


def _normalise(data: iter) -> list[float]:
    min_, max_ = min(data), max(data)
    return [(i - min_) / (max_ - min_) for i in data]
