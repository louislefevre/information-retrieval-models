from collections import Counter
from math import log

import numpy as np
import numpy.linalg as npl

from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.models.Model import Model


class VectorSpace(Model):
    def __init__(self, index: 'InvertedIndex'):
        super().__init__(index)
        self._collection_length = index.collection_length
        self._vocab = sorted(index.vocab)
        self._vocab_count = index.vocab_count

    def _score_query(self, query_tokens: list[str], passages: list[int]) -> dict[int, float]:
        passage_vectors = self._vectorise_passages(passages)
        query_vectors = self._vectorise_query(query_tokens)

        passage_scores = {}
        for pid, vector in passage_vectors.items():
            passage_scores[pid] = self._similarity(query_vectors, vector)
        return passage_scores

    def _vectorise_passages(self, passages: list[int]) -> dict[int, np.ndarray]:
        passage_vectors = {}
        for term, inv_list in self._index.items():
            index = self._vocab.index(term)
            for pid, posting in inv_list.postings.items():
                if pid not in passages:
                    continue
                if pid not in passage_vectors:
                    passage_vectors[pid] = np.zeros(self._vocab_count)
                passage_vectors[pid][index] = posting.tfidf
        return passage_vectors

    def _vectorise_query(self, query_words: list[str]) -> np.ndarray:
        query_vectors = np.zeros(self._vocab_count)
        counter = Counter(query_words)
        words_count = len(query_words)
        for word in query_words:
            if word not in self._index:
                continue
            index = self._vocab.index(word)
            tf = counter[word] / words_count
            df = self._index[word].doc_freq
            idf = log(self._collection_length / df)
            query_vectors[index] = (tf / df) * idf

        return query_vectors

    @staticmethod
    def _similarity(query_vectors: np.ndarray, passage_vectors: np.ndarray) -> float:
        dot_product = np.dot(query_vectors, passage_vectors)
        norms = npl.norm(query_vectors) * npl.norm(passage_vectors)
        return dot_product / norms
