import argparse

from retrieval.DatasetParser import DatasetParser
from retrieval.util.FileManager import write_txt


def main():
    parser = argparse.ArgumentParser(description='Information retrieval models.')
    parser.add_argument('dataset', help='dataset for retrieving passages and queries')
    parser.add_argument('model', help='model for ranking passages against queries')
    parser.add_argument('-p', '--plot', action='store_true', help='generate term frequency graph')
    parser.add_argument('-s', '--smoothing', help='smoothing for the Query Likelihood model')

    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    smoothing = args.smoothing
    plot = args.plot
    parser = DatasetParser(dataset)
    results = parser.parse(model, plot_freq=plot, smoothing=smoothing)

    models = {'bm25': 'BM25', 'vector': 'VS', 'query': 'LM'}
    smoothers = {'laplace': '-Laplace', 'lidstone': '-Lidstone', 'dirichlet': '-Dirichlet'}
    model = models[model]
    smoothing = smoothers[smoothing] if smoothing is not None else ""

    data = ''
    for qid, passages in results.items():
        for rank, (pid, score) in enumerate(passages.items()):
            data += f"{qid}\t{'A1'}\t{pid}\t{rank}\t{format(score, '.2f')}\t{model}{smoothing}\n"

    write_txt(f'results/{model}.txt', data)


if __name__ == '__main__':
    main()
