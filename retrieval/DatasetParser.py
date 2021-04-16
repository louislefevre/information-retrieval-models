import os

from inverted_index import InvertedIndex
from retrieval.data.Dataset import Dataset
from retrieval.models.BM25 import BM25
from retrieval.models.QueryLikelihood import QueryLikelihood
from retrieval.models.VectorSpace import VectorSpace
from retrieval.util.FileManager import read_pickle, write_pickle
from util.TextProcessor import clean


class DatasetParser:
    def __init__(self, dataset_path: str):
        self._dataset = Dataset(dataset_path)
        self._passages = self._dataset.passages()
        self._queries = self._dataset.queries()
        self._mapping = self._dataset.id_mapping()

    def parse(self, model: str, smoothing: str = None, index_path: str = 'index.p') -> dict[str, dict[str, float]]:
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
    def _generate_index(file: str, passages: dict[str, str]) -> InvertedIndex:
        if os.path.isfile(file) and not os.stat(file).st_size == 0:
            print(f"Generating index from '{file}'...")
            return read_pickle(file)
        print(f"Generating index - this will take a few minutes...")
        index = InvertedIndex()
        for doc_name, content in passages.items():
            tokens = clean(content)
            index.add_document(str(doc_name), tokens)

        write_pickle(index, file)
        return index
