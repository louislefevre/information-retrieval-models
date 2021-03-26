import os

from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.models.BM25 import BM25
from retrieval.models.QueryLikelihood import QueryLikelihood
from retrieval.models.VectorSpace import VectorSpace
from retrieval.data.Dataset import Dataset
from retrieval.util.FileManager import write_pickle, read_pickle


def display_results(query: str, results: dict[int, float], passages: dict[int, str]):
    print(query)
    count = 0
    for pid, score in results.items():
        if count >= 100:
            break
        print(f"{pid}({round(score, 2)}): {passages[pid]}")
        count += 1
    print()


def generate_index(file: str, passages: dict[int, str]):
    if os.path.isfile(file) and not os.stat(file).st_size == 0:
        inverted_index = read_pickle(file)
    else:
        inverted_index = InvertedIndex(passages)
        inverted_index.parse()
        write_pickle(inverted_index, file)

    return inverted_index


def main():
    dataset = Dataset('dataset/candidate_passages_top1000.tsv')
    passages = dataset.passages()
    queries = dataset.queries()
    mapping = dataset.id_mapping()

    index = generate_index('index.p', passages)
    model = BM25(index, mapping)

    for qid, query in queries.items():
        results = model.rank(qid, query)
        display_results(query, results, passages)


if __name__ == '__main__':
    main()
