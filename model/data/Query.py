from dataclasses import dataclass

from model.util.FileReader import read


@dataclass
class Query:
    qid: int
    query: str

    @staticmethod
    def convert(file_name: str) -> list:
        rows = read(file_name)
        return [Query(int(row[0]), row[1]) for row in rows]
