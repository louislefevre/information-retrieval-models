import math
import numpy as np

from inverted_index import InvertedIndex
from retrieval.models.Model import Model


class QueryLikelihood(Model):
    def __init__(self, index: InvertedIndex, mapping: dict[str, list[str]], smoothing: str):
        super().__init__(index, mapping)
        self._all_words = index.words()
        self._word_count = index.word_count()
        self._vocab_count = index.vocab_count()
        self._counter = index.word_counter()
        self._smoothing = smoothing

    def _score_passage(self, pid: str, query_words: list[str]) -> float:
        probabilities = []
        for word in query_words:
            probabilities.append(self._probability(pid, word))
        try:
            return math.log(np.prod(probabilities))
        except ValueError:
            return 0.0

    def _probability(self, pid: str, word: str) -> float:
        tf = self._index.get(word).frequency(pid) if word in self._index.documents[pid] else 0
        dl = self._index.word_count(doc_name=pid)
        v = self._vocab_count

        if self._smoothing == 'laplace':
            return (tf + 1) / (dl + v)
        elif self._smoothing == 'lidstone':
            return (tf + 0.5) / (dl + (0.5 * v))
        elif self._smoothing == 'dirichlet':
            return (dl / (dl + 2000)) * (tf / dl) + (2000 / (dl + 2000)) * \
                   (self._counter[word] / self._word_count)
        else:
            raise RuntimeError("Invalid smoothing - select 'laplace', 'lidstone', or 'dirichlet'.")
