import os

from retrieval.data.Dataset import Dataset
from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.models.BM25 import BM25
from retrieval.models.QueryLikelihood import QueryLikelihood
from retrieval.models.VectorSpace import VectorSpace
from retrieval.util.FileManager import read_pickle, write_pickle


class DatasetParser:
    def __init__(self, dataset: 'Dataset'):
        self._dataset = dataset

    def parse(self, model: str, smoothing: str = None) -> dict[int, dict[int, float]]:
        index = self._generate_index('index.p', self._dataset.passages())
        mapping = self._dataset.id_mapping()

        if model == 'bm25':
            model = BM25(index, mapping)
        elif model == 'vs':
            model = VectorSpace(index, mapping)
        elif model == 'lm':
            model = QueryLikelihood(index, mapping, smoothing)
        else:
            raise ValueError("Invalid retrieval model - select 'bm25', 'vs', or 'lm'.")

        print("Ranking queries against passages...")
        return {qid: model.rank(qid, query) for qid, query in self._dataset.queries().items()}

    @staticmethod
    def _generate_index(file_name: str, passages: dict[int, str]) -> InvertedIndex:
        if os.path.isfile(file_name) and not os.stat(file_name).st_size == 0:
            print(f"Generating index from '{file_name}'...")
            return read_pickle(file_name)
        print(f"Generating index - this will take a few minutes...")
        inverted_index = InvertedIndex(passages)
        inverted_index.parse()
        write_pickle(inverted_index, file_name)
        return inverted_index
