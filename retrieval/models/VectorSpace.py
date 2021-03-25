import math
from math import log10

from numpy import dot
from numpy.linalg import norm

from retrieval.InvertedIndex import InvertedIndex
from retrieval.models.Model import Model
from retrieval.util.TextProcessor import clean


class VectorSpace(Model):
    def __init__(self, index: 'InvertedIndex'):
        super().__init__(index)
        self._collection_length = index.collection_length
        self._vocab = index.vocab
        self._vocab_count = index.vocab_count

    def _parse_query(self, query: str) -> dict[int, float]:
        passage_vectors, query_vectors = self._vectorise(query)
        passage_scores = {}

        for pid, vectors in passage_vectors.items():
            passage_scores[pid] = self._cos_sim(query_vectors, vectors)

        return passage_scores

    def _vectorise(self, query: str) -> tuple[dict[int, list[float]], list[float]]:
        query_keywords = [word for word in clean(query)]
        passage_vectors = {}
        query_vectors = [0.0] * self._vocab_count

        for word in query_keywords:
            if word not in self._index:
                continue
            index = self._vocab.index(word)

            # Query vectorisation
            tf = query.count(word) / self._index[word].doc_freq
            df = self._index[word].doc_freq
            idf = log10(self._collection_length / df)
            query_vectors[index] = tf * idf

            # Passage vectorisation
            for pid in self._index[word].postings.keys():
                if pid not in passage_vectors:
                    passage_vectors[pid] = [0.0] * self._vocab_count
                tfidf = self._index[word].get_posting(pid).tfidf
                passage_vectors[pid][index] = tfidf

        return passage_vectors, query_vectors

    @staticmethod
    def _cos_sim(query_vectors: list[float], passage_vectors: list[float]) -> float:
        dot_product = dot(query_vectors, passage_vectors)
        norms = norm(query_vectors) * norm(passage_vectors)
        return dot_product / norms
        # return spatial.distance.cosine(query_vectors, passage_vectors)

    @staticmethod
    def cosine_similarity(v1, v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy / math.sqrt(sumxx * sumyy)
