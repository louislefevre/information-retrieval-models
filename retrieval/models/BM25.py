from math import log

from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.models.Model import Model


class BM25(Model):
    def __init__(self, index: 'InvertedIndex'):
        super().__init__(index)
        self._collection_length = index.collection_length
        self._avg_length = index.avg_length

    def _parse_query(self, query_words: list[str]) -> dict[int, float]:
        passage_scores = {}
        for word in query_words:
            if word not in self._index:
                continue
            inv_list = self._index[word]
            for pid in inv_list.postings.keys():
                if pid in passage_scores:
                    passage_scores[pid] += self._score_passage(pid, word)
                else:
                    passage_scores[pid] = self._score_passage(pid, word)
        return passage_scores

    def _score_passage(self, pid: int, word: str) -> float:
        # Constants
        k1 = 1.2
        b = 0.75

        # Parameters
        inv_index = self._index[word]
        N = self._collection_length
        n = inv_index.doc_freq
        f = inv_index.get_posting(pid).freq
        dl = float(len(self._collection[pid]))
        avg_dl = float(self._avg_length)

        # Formulas
        K = k1 * ((1 - b) + b * (dl / avg_dl))
        expr_1 = log(((N - n + 0.5) / (n + 0.5)) + 1)
        expr_2 = ((f * (k1 + 1)) / (f + K))
        return expr_1 * expr_2
