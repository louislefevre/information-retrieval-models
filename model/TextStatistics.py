from matplotlib import pyplot as plt
from nltk import FreqDist
from sklearn.preprocessing import normalize

from model.util.FileReader import process_passage_collection
from model.util.TextProcessor import clean


def plot():
    collection = process_passage_collection()
    passages = [clean(passage, remove_sw=False) for passage in collection.values()]
    tokens = [token for tokens in passages for token in tokens]
    _plot_frequencies(tokens)


def _plot_frequencies(tokens):
    dist = FreqDist(tokens)
    common = dist.most_common(100)
    frequencies = [tup[1] for tup in common]
    frequencies = normalize([frequencies])[0]
    frequencies = [prob * 0.1 for prob in frequencies]
    _generate_figure(frequencies, title="Term Frequency", x_label="Rank", y_label="Probability",
                     save=True)


def _generate_figure(data, title=None, x_label=None, y_label=None, save=False, show=False):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save:
        plt.savefig("term-frequencies.png")
    if show:
        plt.show()
