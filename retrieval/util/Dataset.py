from retrieval.util.FileManager import read_tsv


class Dataset:
    def __init__(self, file_name: str):
        self._rows = read_tsv(file_name)

    def qid_pid(self) -> dict[int, list[int]]:
        mapping = {}
        for row in self._rows:
            qid, pid = int(row[0]), int(row[1])
            if qid not in mapping:
                mapping[qid] = []
            mapping[qid] += pid
        return mapping

    def queries(self) -> dict[int, str]:
        return {int(row[0]): row[2] for row in self._rows}

    def passages(self) -> dict[int, str]:
        return {int(row[1]): row[3] for row in self._rows}