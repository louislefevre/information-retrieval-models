from dataclasses import dataclass, field

from retrieval.util.FileManager import read_tsv


class Dataset:
    def __init__(self, file_name: str):
        self._file_name = file_name
        self._queries: dict[int, 'Query'] = {}

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

        self._queries = queries

    def id_mapping(self) -> dict[int, list[int]]:
        return {query.qid: [pid for pid in query.passages] for query in self._queries.values()}

    def relevant_mapping(self) -> dict[int, list[int]]:
        return {qid: query.relevant for qid, query in self._queries.items()}

    def queries(self) -> dict[int, str]:
        return {query.qid: query.text for query in self._queries.values()}

    def passages(self) -> dict[int, str]:
        return {pid: text for query in self._queries.values() for pid, text in query.passages.items()}


@dataclass
class Query:
    qid: int
    text: str
    passages: dict[int, str] = field(default_factory=dict)
    relevant: list[int] = field(default_factory=list)

    def add_passage(self, pid: int, text: str, relevance: float):
        self.passages[pid] = text
        if relevance > 0.0:
            self.relevant.append(pid)
