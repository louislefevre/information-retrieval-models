# Text Statistics (20 marks).
# - Perform any type of pre-processing on the collection as you think is required.
# - Implement a function that counts the frequency of terms from the provided dataset, plot the
#   distribution of term frequencies and verify if they follow Zipf’s law.
# - Report the values of the parameters for Zipf’s law for this collection. You need to use the
#   full collection (file named passage_collection.txt) for this question.
# - Generate a plot that shows how the results you get using the model based on Zipf’s law compare
#   with the values you get from the actual collection.

from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from nltk import FreqDist


def plot_frequencies(tokens):
    freq = FreqDist(tokens)
    common = freq.most_common(100)
    frequencies = [tup[1] for tup in common]
    frequencies = normalize([frequencies])[0]
    plot(frequencies, title="Term Frequency", x_label="Rank", y_label="Probability", show=True)


def plot(data, title=None, x_label=None, y_label=None, save=False, show=False):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save:
        plt.savefig("term-frequencies.png")
    if show:
        plt.show()
