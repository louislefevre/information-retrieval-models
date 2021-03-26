import math

import numpy as np

from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.models.Model import Model


class QueryLikelihood(Model):
    def __init__(self, index: 'InvertedIndex', mapping: dict[int, list[int]]):
        super().__init__(index, mapping)
        self._vocab_count = index.vocab_count
        self._all_words = index.words
        self._word_count = index.word_count
        self._counter = index.counter

    def _score_query(self, query_words: list[str], passages: list[int]) -> dict[int, float]:
        language_models = {}
        for pid in passages:
            probabilities = []
            for word in query_words:
                probabilities.append(self._probability(pid, word, smoothing='dirichlet'))
            language_models[pid] = math.log(np.prod(probabilities))

        return language_models

    def _probability(self, pid: int, word: str, smoothing=None) -> float:
        tf = self._index[word].get_posting(pid).freq if word in self._collection[pid] else 0
        dl = len(self._collection[pid])
        v = self._vocab_count

        if smoothing is None:
            return tf / dl
        elif smoothing == 'laplace':
            return (tf + 1) / (dl + v)
        elif smoothing == 'lidstone':
            return (tf + 0.5) / (dl + (0.5 * v))
        elif smoothing == 'dirichlet':
            return (dl / (dl + 2000)) * (tf / dl) + (2000 / (dl + 2000)) * \
                   (self._counter[word] / self._word_count)
        else:
            raise RuntimeError("Invalid smoothing.")
