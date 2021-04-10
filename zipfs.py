import argparse
import itertools
from collections import Counter

from matplotlib import pyplot as plt
from tabulate import tabulate

from data.Dataset import Dataset
from util.FileManager import write_txt
from util.TextProcessor import clean_collection


def _zipfs_distribution(counter: Counter):
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


def _plot_distribution(prob_dist: list[float]):
    zipf_dist = [0.1 / i for i in range(1, 101)]
    _generate_figure(prob_dist, zipf_dist, title="Zipf's Law", x_label="Rank",
                     y_label="Probability", file_name='zipf-plot.png')


def _report_parameters(rows: list[list], c: float):
    table = tabulate(rows, headers=['Word', 'Freq', 'r', 'Pr', 'r*Pr'])
    c = round(c / len(rows), 3)
    data = table + f'\n\nc = {c}'
    file_name = 'zipf-parameters.txt'
    write_txt(file_name, data)
    print(f'{file_name} was created.')


def _generate_figure(*data, title=None, x_label=None, y_label=None, file_name='figure.png'):
    for d in data:
        plt.plot(d)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(file_name)
    print(f'{file_name} was created.')


def main():
    parser = argparse.ArgumentParser(description='Zipfs law distribution plot and report.')
    parser.add_argument('dataset', help='dataset for retrieving passages')
    args = parser.parse_args()

    print("This script is only for generating a distribution plot and parameter report for Zipfs law."
          "To run the IR models program, execute 'start.py'.")

    print("Processing dataset - this will take a few minutes...")
    dataset = Dataset(args.dataset)
    collection = clean_collection(dataset.passages(), remove_sw=False)

    print("Analysing data...")
    words = list(itertools.chain.from_iterable(collection.values()))
    counter = Counter(words)

    print("Generating plot...")
    _zipfs_distribution(counter)


if __name__ == '__main__':
    main()
