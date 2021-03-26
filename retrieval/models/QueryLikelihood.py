import itertools
import math
from collections import Counter

import numpy as np

from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.models.Model import Model


class QueryLikelihood(Model):
    def __init__(self, index: 'InvertedIndex'):
        super().__init__(index)
        self._vocab_count = index.vocab_count
        self._all_words = list(itertools.chain.from_iterable(self._collection.values()))
        self._word_count = len(self._all_words)
        self._counter = Counter(self._all_words)

    def _parse_query(self, query_words: list[str]) -> dict[int, float]:
        relevant_passages = set()
        for word in query_words:
            if word not in self._index:
                continue
            for pid in self._index[word].postings:
                relevant_passages.add(pid)

        language_models = {}
        for pid in relevant_passages:
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
