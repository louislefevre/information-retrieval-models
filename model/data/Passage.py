from dataclasses import dataclass

from model.util.FileReader import read


@dataclass
class Passage:
    pid: int
    query: str

    @staticmethod
    def convert(file_name: str) -> list:
        rows = read(file_name)
        return [Passage(int(row[0]), row[1]) for row in rows]
