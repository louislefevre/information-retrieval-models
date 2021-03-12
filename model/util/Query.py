from typing import Union

from model.util.FileReader import read


class Query:
    def __init__(self, qid: Union[str, int], query: str):
        self._qid = int(qid)
        self._query = query

    @property
    def qid(self) -> int:
        return self._qid

    @property
    def query(self) -> str:
        return self._query

    @staticmethod
    def convert(file_name: str) -> list:
        rows = read(file_name)
        return [Query(row[0], row[1]) for row in rows]
