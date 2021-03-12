from typing import Union

from model.util.FileReader import read


class Passage:
    def __init__(self, pid: Union[str, int], passage: str):
        self._pid = int(pid)
        self._passage = passage

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def passage(self) -> str:
        return self._passage

    @staticmethod
    def convert(file_name: str) -> list:
        rows = read(file_name)
        return [Passage(row[0], row[1]) for row in rows]
