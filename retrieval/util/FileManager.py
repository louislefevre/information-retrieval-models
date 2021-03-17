import csv
from dataclasses import dataclass
from typing import Union, Tuple
import pickle


def read(file_name: str) -> Union[list, str]:
    file = open(file_name)
    if file_name.endswith('.tsv'):
        return list(csv.reader(file, delimiter="\t"))
    elif file_name.endswith('.txt'):
        return file.read()
    else:
        raise RuntimeError("Invalid file type: only '.tsv' and '.txt' files are accepted.")


def write_pickle(data: object, file_name: str):
    pickle.dump(data, open(file_name, "wb"))


def read_pickle(file_name: str) -> object:
    return pickle.load(open(file_name, "rb"))


def process_candidate_passages_and_queries() -> Tuple[dict[int, str], dict[int, str]]:
    candidate_passages = "dataset/candidate_passages_top1000.tsv"
    rows = read(candidate_passages)
    passages, queries = {}, {}
    for row in rows:
        qid, query = int(row[0]), row[2]
        pid, passage = int(row[1]), row[3]
        queries[qid] = query
        passages[pid] = passage
    return passages, queries


def process_relations() -> dict[int, list[int]]:
    candidate_passages = "dataset/candidate_passages_top1000.tsv"
    rows = read(candidate_passages)
    relations = {}
    for row in rows:
        qid, pid = int(row[0]), int(row[1])
        if qid in relations:
            relations[qid] += [pid]
        else:
            relations[qid] = [pid]
    return relations


def process_test_queries() -> dict[int, str]:
    test_queries = "dataset/test-queries.tsv"
    rows = read(test_queries)
    return {row[0]: row[1] for row in rows}


def process_passage_collection() -> dict[int, str]:
    passage_collection = "dataset/passage_collection_new.txt"
    rows = read(passage_collection).splitlines()
    return {pid: passage for pid, passage in enumerate(rows)}


def process_queries() -> list['Query']:
    candidate_passages = "dataset/candidate_passages_top1000.tsv"
    rows = read(candidate_passages)

    queries = {}
    for row in rows:
        queries[int(row[0])] = row[2]

    query_objs = []
    for qid, query in queries.items():
        query_objs.append(Query(qid, query, {}))

    for query in query_objs:
        for row in rows:
            if int(row[0]) == query.qid:
                query.passages[int(row[1])] = row[3]

    return query_objs


@dataclass
class Query:
    qid: int
    text: str
    passages: dict[int, str]
