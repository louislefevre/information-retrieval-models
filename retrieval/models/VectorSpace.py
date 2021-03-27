from collections import Counter
from math import log

import numpy as np
import numpy.linalg as npl

from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.models.Model import Model


class VectorSpace(Model):
    def __init__(self, index: 'InvertedIndex', mapping: dict[int, list[int]]):
        super().__init__(index, mapping)
        self._vocab = index.vocab
        self._collection_length = index.collection_length
        self._vocab_count = index.vocab_count

    def _score_query(self, query_words: list[str], passages: list[int]) -> dict[int, float]:
        return {pid: self._similarity(pid, query_words) for pid in passages}

    def _similarity(self, pid: int, query_words: list[str]) -> float:
        vocab = list(set(self._collection[pid]))
        vocab_count = len(vocab)

        passage_vector = np.zeros(vocab_count)
        for idx, word in enumerate(vocab):
            passage_vector[idx] = self._index[word].get_posting(pid).tfidf

        query_vector = np.zeros(vocab_count)
        counter = Counter(query_words)
        for word in query_words:
            tfidf = self._tfidf(counter, word)
            if word in vocab:
                idx = vocab.index(word)
                query_vector[idx] = tfidf
            else:
                query_vector = np.append(query_vector, tfidf)
                passage_vector = np.append(passage_vector, 0)

        return self._cos_sim(query_vector, passage_vector)

    def _tfidf(self, counter: Counter, word: str) -> float:
        tf = counter[word] / sum(counter.values())
        df = self._index[word].doc_freq
        idf = log(self._collection_length / df)
        return tf * idf

    @staticmethod
    def _cos_sim(query_vectors: np.ndarray, passage_vectors: np.ndarray) -> float:
        dot_product = np.dot(query_vectors, passage_vectors)
        norms = npl.norm(query_vectors) * npl.norm(passage_vectors)
        return dot_product / norms
