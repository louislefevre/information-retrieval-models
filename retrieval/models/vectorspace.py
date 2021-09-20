from collections import Counter
from math import log

import numpy as np
import numpy.linalg as npl

from retrieval.util.invertedindex import InvertedIndex
from retrieval.models.model import Model


class VectorSpace(Model):
    def __init__(self, index: 'InvertedIndex', mapping: dict[int, list[int]]):
        super().__init__(index, mapping)
        self._vocab = index.vocab
        self._collection_length = index.collection_length
        self._vocab_count = index.vocab_count

    def _score_passage(self, pid: int, query_words: list[str]) -> float:
        return self._similarity(pid, query_words)

    def _similarity(self, pid: int, query_words: list[str]) -> float:
        vocab = list(set(self._collection[pid]))
        vocab_count = len(vocab)

        passage_vector = np.zeros(vocab_count)
        for idx, word in enumerate(vocab):
            passage_vector[idx] = self._index[word].get_posting(pid).tfidf

        query_vector = np.zeros(vocab_count)
        counter = Counter(query_words)
        max_freq = counter.most_common(1)[0][1]
        for word in query_words:
            if word not in self._index:
                continue
            tf = (0.5 + (0.5 * (counter[word] / max_freq)))
            idf = log(self._collection_length / self._index[word].doc_freq)
            tfidf = tf * idf
            if word in vocab:
                idx = vocab.index(word)
                query_vector[idx] = tfidf
            else:
                query_vector = np.append(query_vector, tfidf)
                passage_vector = np.append(passage_vector, 0)

        return self._cos_sim(query_vector, passage_vector)

    @staticmethod
    def _cos_sim(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
        dot_product = np.dot(vector_1, vector_2)
        norms = npl.norm(vector_1) * npl.norm(vector_2)
        return dot_product / norms
