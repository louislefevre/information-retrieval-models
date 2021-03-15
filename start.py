import os

from model.InvertedIndex import InvertedIndex
from model.util.FileManager import process_test_queries, process_candidate_passages_and_queries, \
    process_passage_collection, write_pickle, read_pickle


def generate_index():
    test_queries = process_test_queries()
    candidate_passages, candidate_queries = process_candidate_passages_and_queries()
    passage_collection = process_passage_collection()
    inverted_index = InvertedIndex(passage_collection)
    inverted_index.index_collection()
    return inverted_index


def main():
    index_file = 'index.p'
    if os.path.isfile(index_file) and not os.stat(index_file).st_size == 0:
        index = read_pickle(index_file)
    else:
        index = generate_index()
        write_pickle(index, index_file)


if __name__ == '__main__':
    main()
