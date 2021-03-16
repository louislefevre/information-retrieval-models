import os

from model.BM25 import BM25
from model.InvertedIndex import InvertedIndex
from model.util.FileManager import process_candidate_passages_and_queries, write_pickle, \
    read_pickle


def display_results(query: str, results: dict[int, float], passages: dict[int, str]):
    print(query)
    for pid, score in results.items():
        print(f"{pid}({round(score, 2)}): {passages[pid]}")


def generate_index(passages):
    inverted_index = InvertedIndex(passages)
    inverted_index.index_collection()
    return inverted_index


def main():
    passages, queries = process_candidate_passages_and_queries()
    index_file = 'index.p'

    if os.path.isfile(index_file) and not os.stat(index_file).st_size == 0:
        index = read_pickle(index_file)
    else:
        index = generate_index(passages)
        write_pickle(index, index_file)

    bm25 = BM25(index)
    for query in queries.values():
        results = bm25.rank(query)
        display_results(query, results, passages)


if __name__ == '__main__':
    main()
