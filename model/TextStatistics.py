from matplotlib import pyplot as plt
from nltk import FreqDist
from sklearn.preprocessing import normalize

from model.util.FileReader import read
from model.util.TextProcessor import tokenize, normalise, stem


def plot():
    passage_collection = "dataset/passage_collection_new.txt"
    text = read(passage_collection)
    tokens = tokenize(text)
    tokens = normalise(tokens)
    tokens = stem(tokens)
    _plot_frequencies(tokens)


def _plot_frequencies(tokens):
    dist = FreqDist(tokens)
    common = dist.most_common(100)
    frequencies = [tup[1] for tup in common]
    frequencies = normalize([frequencies])[0]
    frequencies = [e*0.1 for e in frequencies]
    _plot_figure(frequencies, title="Term Frequency", x_label="Rank", y_label="Probability")


def _plot_figure(data, title=None, x_label=None, y_label=None, save=False, show=False):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save:
        plt.savefig("term-frequencies.png")
    if show:
        plt.show()
