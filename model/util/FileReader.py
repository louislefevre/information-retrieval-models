import csv
from typing import Union, Tuple

from model.data.Data import Passage, Query


def read(file_name: str) -> Union[list, str]:
    file = open(file_name)
    if file_name.endswith('.tsv'):
        return list(csv.reader(file, delimiter="\t"))
    elif file_name.endswith('.txt'):
        return file.read()[:10000]
    else:
        raise RuntimeError("Invalid file type: only '.tsv' and '.txt' files are accepted.")


def process_candidate_passages_and_queries() -> Tuple[list[Passage], list[Query]]:
    candidate_passages = "dataset/candidate_passages_top1000.tsv"
    rows = read(candidate_passages)
    passages = [Passage(int(row[1]), row[3]) for row in rows]
    queries = [Query(int(row[0]), row[2]) for row in rows]
    return passages, queries


def process_test_queries() -> list[Query]:
    test_queries = "dataset/test-queries.tsv"
    rows = read(test_queries)
    return [Query(int(row[0]), row[1]) for row in rows]


def process_passage_collection() -> list[Passage]:
    passage_collection = "dataset/passage_collection_new.txt"
    rows = read(passage_collection).splitlines()
    return [Passage(i, rows[i]) for i in range(len(rows))]
