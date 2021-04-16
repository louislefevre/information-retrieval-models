from retrieval.util.FileManager import read_tsv


class Dataset:
    def __init__(self, file_name: str):
        self._rows = read_tsv(file_name)

    def id_mapping(self) -> dict[str, list[str]]:
        mapping = {}
        for row in self._rows:
            qid, pid = row[0], row[1]
            if qid not in mapping:
                mapping[qid] = []
            mapping[qid] += [pid]
        return mapping

    def queries(self) -> dict[str, str]:
        return {int(row[0]): row[2] for row in self._rows}

    def passages(self) -> dict[str, str]:
        return {int(row[1]): row[3] for row in self._rows}
