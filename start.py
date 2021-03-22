import os

from retrieval.models.BM25 import BM25
from retrieval.InvertedIndex import InvertedIndex
from retrieval.models.VectorSpace import VectorSpace
from retrieval.util.FileManager import write_pickle, read_pickle, process_queries, \
    process_candidate_passages_and_queries


def display_results(query: str, results: dict[int, float], passages: dict[int, str]):
    print(query)
    for pid, score in results.items():
        print(f"{pid}({round(score, 2)}): {passages[pid]}")
    print()


def generate_index(file: str, passages: dict[int, str]):
    if os.path.isfile(file) and not os.stat(file).st_size == 0:
        inverted_index = read_pickle(file)
    else:
        inverted_index = InvertedIndex(passages)
        inverted_index.parse()
        write_pickle(inverted_index, file)

    return inverted_index


def run_all():
    passages, queries = process_candidate_passages_and_queries()
    index = generate_index('indexes/index.p', passages)
    model = BM25(index)
    for qid, query in queries.items():
        results = model.rank(query, top_n=100)
        display_results(query, results, passages)


def run_isolated():
    queries = process_queries()
    for query in queries:
        index = generate_index(f'indexes/{query.qid}.p', query.passages)
        model = BM25(index)
        results = model.rank(query.text, top_n=100)
        display_results(query.text, results, query.passages)


def main():
    run_all()


if __name__ == '__main__':
    main()
