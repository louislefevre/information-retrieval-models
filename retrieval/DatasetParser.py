import os

from retrieval.util.InvertedIndex import InvertedIndex
from retrieval.models.BM25 import BM25
from retrieval.models.QueryLikelihood import QueryLikelihood
from retrieval.models.VectorSpace import VectorSpace
from retrieval.util.FileManager import read_pickle, write_pickle, read_tsv


class Dataset:
    def __init__(self, file_name: str):
        self._rows = read_tsv(file_name)

    def id_mapping(self) -> dict[int, list[int]]:
        mapping = {}
        for row in self._rows:
            qid, pid = int(row[0]), int(row[1])
            if qid not in mapping:
                mapping[qid] = []
            mapping[qid] += [pid]
        return mapping

    def queries(self) -> dict[int, str]:
        return {int(row[0]): row[2] for row in self._rows}

    def passages(self) -> dict[int, str]:
        return {int(row[1]): row[3] for row in self._rows}


class DatasetParser:
    def __init__(self, dataset_path: str):
        self._dataset = Dataset(dataset_path)
        self._passages = self._dataset.passages()
        self._queries = self._dataset.queries()
        self._mapping = self._dataset.id_mapping()

    def parse(self, model: str, smoothing: str = None, index_path: str = 'index.p') -> dict[int, dict[int, float]]:
        index = self._generate_index(index_path, self._passages)

        if model == 'bm25':
            model = BM25(index, self._mapping)
        elif model == 'vs':
            model = VectorSpace(index, self._mapping)
        elif model == 'lm':
            model = QueryLikelihood(index, self._mapping, smoothing)
        else:
            raise ValueError("Invalid retrieval model - select 'bm25', 'vs', or 'lm'.")

        print("Ranking queries against passages...")
        return {qid: model.rank(qid, query) for qid, query in self._queries.items()}

    @staticmethod
    def _generate_index(file: str, passages: dict[int, str]) -> InvertedIndex:
        if os.path.isfile(file) and not os.stat(file).st_size == 0:
            print(f"Generating index from '{file}'...")
            return read_pickle(file)
        print(f"Generating index - this will take a few minutes...")
        inverted_index = InvertedIndex(passages)
        inverted_index.parse()
        write_pickle(inverted_index, file)
        return inverted_index
