from math import log

from retrieval.InvertedIndex import InvertedIndex
from retrieval.models.Model import Model


class BM25(Model):
    def __init__(self, index: 'InvertedIndex'):
        super().__init__(index)
        self._collection_length = index.collection_length
        self._avg_length = index.avg_length

    def _score_passage(self, pid: int, word: str) -> float:
        # Constants
        r = 0
        R = 0.0
        k1 = 1.2
        k2 = 100.0
        b = 0.75
        qf = 1

        # Parameters
        inv_index = self._index[word]
        n = inv_index.doc_freq
        N = self._collection_length
        dl = float(len(self._collection[pid]))
        avg_dl = float(self._avg_length)
        f = inv_index.get_posting(pid).freq
        K = k1 * ((1 - b) + b * (dl / avg_dl))

        # Formulas
        expr_1 = log(((r + 0.5) / (R - r + 0.5)) /
                     ((n - r + 0.5) / (N - n - R + r + 0.5)))
        expr_2 = ((k1 + 1) * f) / (K + f)
        expr_3 = ((k2 + 1) * qf) / (k2 + qf)
        return expr_1 * expr_2 * expr_3
