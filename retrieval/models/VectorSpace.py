from collections import Counter
from math import log

import numpy as np
from numpy import dot
from numpy.linalg import norm

from retrieval.InvertedIndex import InvertedIndex
from retrieval.models.Model import Model


class VectorSpace(Model):
    def __init__(self, index: 'InvertedIndex'):
        super().__init__(index)
        self._collection_length = index.collection_length
        self._vocab = sorted(index.vocab)
        self._vocab_count = index.vocab_count

    def _parse_query(self, query_words: list[str]) -> dict[int, float]:
        passage_vectors = self._vectorise_passages(query_words)
        query_vectors = self._vectorise_query(query_words)

        passage_scores = {}
        for pid, vector in passage_vectors.items():
            passage_scores[pid] = self._similarity(query_vectors, vector)
        return passage_scores

    def _vectorise_passages(self, query_words: list[str]) -> dict[int, np.ndarray]:
        relevant_passages = set()
        for word in query_words:
            if word not in self._index:
                continue
            for pid in self._index[word].postings:
                relevant_passages.add(pid)

        passage_vectors = {}
        count = 0
        for term, inv_list in self._index.items():
            count += 1
            print(f"{count} / {len(self._index)}")
            index = self._vocab.index(term)
            for pid, posting in inv_list.postings.items():
                if pid not in relevant_passages:
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

            # Query vectorisation
            tf = counter[word] / words_count
            df = self._index[word].doc_freq
            idf = log(self._collection_length / df)
            query_vectors[index] = (tf / df) * idf

        return query_vectors

    @staticmethod
    def _similarity(query_vectors: np.ndarray, passage_vectors: np.ndarray) -> float:
        dot_product = dot(query_vectors, passage_vectors)
        norms = norm(query_vectors) * norm(passage_vectors)
        return dot_product / norms
