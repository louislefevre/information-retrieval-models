import os

from retrieval.data.Dataset import Dataset
from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.models.BM25 import BM25
from retrieval.models.QueryLikelihood import QueryLikelihood
from retrieval.models.VectorSpace import VectorSpace
from retrieval.util.FileManager import read_pickle, write_pickle


class DatasetParser:
    def __init__(self, dataset_path: str):
        self._dataset = self._generate_dataset('dataset.p', dataset_path)
        self._passages = self._dataset.passages()
        self._queries = self._dataset.queries()
        self._mapping = self._dataset.id_mapping()

    def parse(self, model: str, smoothing: str = None) -> dict[int, dict[int, float]]:
        index = self._generate_index('index.p', self._passages)

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
    def _generate_dataset(file_name: str, dataset_name: str):
        if os.path.isfile(file_name) and not os.stat(file_name).st_size == 0:
            print(f"Generating dataset from '{file_name}'...")
            return read_pickle(file_name)
        print(f"Generating dataset - this will take a few minutes...")
        dataset = Dataset(dataset_name)
        dataset.parse()
        write_pickle(dataset, file_name)
        return dataset

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
