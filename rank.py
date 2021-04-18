import argparse

from data.Dataset import Dataset
from retrieval.DatasetParser import DatasetParser
from retrieval.util.FileManager import write_txt


def main():
    parser = argparse.ArgumentParser(description='Information retrieval models.')
    parser.add_argument('dataset', help='dataset for retrieving passages and queries')
    parser.add_argument('model', help='model for ranking passages against queries')
    parser.add_argument('-s', '--smoothing', help='smoothing for the Query Likelihood model')
    parser.add_argument('-l', '--limit', type=int, help='maximum number of results to return for each query')

    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    smoothing = args.smoothing
    limit = args.limit

    if model == 'lm' and smoothing is None:
        raise ValueError("Smoothing must be supplied when using the Query Likelihood model.")
    if not model == 'lm' and smoothing is not None:
        raise ValueError("Smoothing can only be applied to the Query Likelihood model.")

    dataset = Dataset(dataset)
    parser = DatasetParser(dataset)

    results = parser.parse(model, smoothing=smoothing)
    if limit:
        results = {qid: {r: ranks[r] for r in list(ranks)[:limit]} for qid, ranks in results.items()}

    model = model.upper()
    smoothing = f'-{smoothing.capitalize()}' if smoothing is not None else ""
    data = ''
    for qid, passages in results.items():
        for rank, (pid, score) in enumerate(passages.items()):
            data += f"{qid}\t{'A1'}\t{pid}\t{rank+1}\t{format(score, '.2f')}\t{model}{smoothing}\n"
    write_txt(f'results/{model}{smoothing}.txt', data)


if __name__ == '__main__':
    main()
