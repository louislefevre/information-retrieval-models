import argparse

from data.Dataset import Dataset
from util.FileManager import read_txt
from util.Metrics import mean_average_precision, mean_ndcg


def main():
    parser = argparse.ArgumentParser(description='Information retrieval evaluating.')
    parser.add_argument('dataset', help='dataset for retrieving passages and queries')
    parser.add_argument('results', help='results to evaluate')

    args = parser.parse_args()
    dataset = Dataset(args.dataset)

    results: dict[int, dict[int, float]] = {}
    lines = read_txt(args.results)
    for line in lines:
        line = line.split(sep='\t')
        qid, pid, score = int(line[0]), int(line[2]), float(line[4])
        if qid not in results:
            results[qid] = {}
        results[qid][pid] = score

    print(mean_average_precision(results, dataset.relevant_mapping()))
    print(mean_ndcg(results, dataset.relevant_mapping()))


if __name__ == '__main__':
    main()
