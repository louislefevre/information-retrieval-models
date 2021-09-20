import numpy as np
import numpy.linalg as npl
from collections import Counter
from math import log
from retrieval.invertedindex import InvertedIndex
from retrieval.util.textprocessor import clean


class Model:
    def __init__(self, index: 'InvertedIndex', mapping: dict[int, list[int]]):
        self._collection = index.collection
        self._index = index.index
        self._mapping = mapping

    def rank(self, qid: int, query: str) -> dict[int, float]:
        query_words = self._clean_query(query)
        passages = self._relevant_passages(qid, query_words)

        scores = {}
        for pid in passages:
            scores[pid] = self._score_passage(pid, query_words)

        ranks = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {pid: score for pid, score in ranks}

    def _clean_query(self, query: str) -> list[str]:
        return [word for word in clean(query)]

    def _relevant_passages(self, qid: int, query_words: list[str]) -> list[int]:
        return list(set(pid for word in query_words if word in self._index
                        for pid in self._index[word].postings if pid in self._mapping[qid]))

    def _score_passage(self, pid: int, query_words: list[str]) -> float:
        raise NotImplementedError


class BM25(Model):
    def __init__(self, index: 'InvertedIndex', mapping: dict[int, list[int]]):
        super().__init__(index, mapping)
        self._collection_length = index.collection_length
        self._avg_length = index.avg_length

    def _score_passage(self, pid: int, query_words: list[str]) -> float:
        score = 0
        for word in query_words:
            if word not in self._index or not self._index[word].contains_posting(pid):
                continue
            score += self._score(pid, word)
        return score

    def _score(self, pid: int, word: str) -> float:
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


class QueryLikelihood(Model):
    def __init__(self, index: InvertedIndex, mapping: dict[int, list[int]], smoothing: str):
        super().__init__(index, mapping)
        self._all_words = index.words
        self._word_count = index.word_count
        self._vocab_count = index.vocab_count
        self._counter = index.counter
        self._smoothing = smoothing

    def _score_passage(self, pid: int, query_words: list[str]) -> float:
        probabilities = []
        for word in query_words:
            probabilities.append(self._probability(pid, word))
        try:
            return log(np.prod(probabilities))
        except ValueError:
            return 0.0

    def _probability(self, pid: int, word: str) -> float:
        tf = self._index[word].get_posting(pid).freq if word in self._collection[pid] else 0
        dl = len(self._collection[pid])
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
