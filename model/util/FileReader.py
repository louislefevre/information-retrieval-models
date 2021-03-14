import csv
from typing import Union, Tuple


def read(file_name: str) -> Union[list, str]:
    file = open(file_name)
    if file_name.endswith('.tsv'):
        return list(csv.reader(file, delimiter="\t"))
    elif file_name.endswith('.txt'):
        return file.read()[:10000]
    else:
        raise RuntimeError("Invalid file type: only '.tsv' and '.txt' files are accepted.")


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


def process_test_queries() -> dict[int, str]:
    test_queries = "dataset/test-queries.tsv"
    rows = read(test_queries)
    return {qid: query for qid, query in enumerate(rows)}


def process_passage_collection() -> dict[int, str]:
    passage_collection = "dataset/passage_collection_new.txt"
    rows = read(passage_collection).splitlines()
    return {pid: passage for pid, passage in enumerate(rows)}

