from model.util.FileReader import process_test_queries, process_candidate_passages_and_queries, \
    process_passage_collection
from model.InvertedIndex import InvertedIndex


def main():
    test_queries = process_test_queries()
    candidate_passages, candidate_queries = process_candidate_passages_and_queries()
    passage_collection = process_passage_collection()[:2]

    index = InvertedIndex()
    for passage in passage_collection:
        index.index_passage(passage)


if __name__ == '__main__':
    main()
