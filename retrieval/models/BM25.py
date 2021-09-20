from math import log

from inverted_index import InvertedIndex
from retrieval.models.Model import Model


class BM25(Model):
    def __init__(self, index: 'InvertedIndex', mapping: dict[str, list[str]]):
        super().__init__(index, mapping)
        self._document_count = index.document_count()
        self._avg_length = index.average_length()

    def _score_passage(self, pid: str, query_words: list[str]) -> float:
        score = 0
        for word in query_words:
            if word not in self._index or pid not in self._index.get(word):
                continue
            score += self._score(pid, word)
        return score

    def _score(self, pid: str, word: str) -> float:
        # Constants
        k1 = 1.2
        b = 0.75

        # Parameters
        inv_list = self._index.get(word)
        N = self._document_count
        n = inv_list.document_frequency
        f = inv_list.get(pid).frequency
        dl = float(self._index.word_count(doc_id=pid))
        avg_dl = float(self._avg_length)

        # Formulas
        K = k1 * ((1 - b) + b * (dl / avg_dl))
        expr_1 = log(((N - n + 0.5) / (n + 0.5)) + 1)
        expr_2 = ((f * (k1 + 1)) / (f + K))
        return expr_1 * expr_2
