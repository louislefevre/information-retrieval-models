from dataclasses import dataclass, field

from retrieval.util.FileManager import read_tsv


class Dataset:
    def __init__(self, file_name: str):
        self._file_name = file_name
        self._queries: list['Query'] = []

    def parse(self):
        rows = read_tsv(self._file_name)[1:]
        queries = {}
        for row in rows:
            qid, query_text = int(row[0]), row[2]
            pid, passage_text, relevancy = int(row[1]), row[3], float(row[4])
            if qid not in queries:
                queries[qid] = Query(qid, query_text)
            query = queries[qid]
            query.add_passage(pid, passage_text, relevancy)

        self._queries = list(queries.values())

    def id_mapping(self) -> dict[int, list[int]]:
        return {query.qid: [passage.pid for passage in query.passages] for query in self._queries}

    def queries(self) -> dict[int, str]:
        return {query.qid: query.text for query in self._queries}

    def passages(self) -> dict[int, str]:
        return {passage.pid: passage.text for query in self._queries for passage in query.passages}


@dataclass
class Query:
    qid: int
    text: str
    passages: list['Passage'] = field(default_factory=list)

    def add_passage(self, pid: int, text: str, relevance: float):
        self.passages.append(Passage(pid, text, relevance))


@dataclass
class Passage:
    pid: int
    text: str
    relevance: float
