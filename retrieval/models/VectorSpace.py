from collections import Counter
from math import log

import numpy as np
import numpy.linalg as npl

from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.models.Model import Model


class VectorSpace(Model):
    def __init__(self, index: 'InvertedIndex', mapping: dict[int, list[int]]):
        super().__init__(index, mapping)
        self._collection_length = index.collection_length
        self._vocab = index.vocab
        self._vocab_count = index.vocab_count

    def _score_query(self, query_tokens: list[str], passages: list[int]) -> dict[int, float]:
        #passage_vectors = self._vectorise_passages(passages)
        #query_vectors = self._vectorise_query(query_tokens)
        #passage_vectors, query_vectors = self._vectorise(query_tokens, passages)
        return self._vectorise_passage_length(query_tokens, passages)

        passage_scores = {}
        for pid, vector in passage_vectors.items():
            passage_scores[pid] = self._similarity(query_vectors, vector)
            if pid == 8303857:
                print(f'Passage: {self._collection[pid]}')
                print(f'Query: {query_tokens}')
                print(f'Query Vector: {query_vectors[query_vectors != 0]}')
                print(f'Passage Vector: {vector[vector != 0]}')
                print(f'QVector Length: {len(query_vectors)}')
                print(f'PVector Length: {len(vector)}')

        return passage_scores

    ######## PASSAGE-LENGTH VECTOR ########
    def _vectorise_passage_length(self, query_words: list[str], passages: list[int]) -> dict[int, float]:
        passage_scores = {}
        for pid in passages:
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

            passage_scores[pid] = self._similarity(query_vector, passage_vector)

        return passage_scores

    def _tfidf(self, counter: Counter, word: str) -> float:
        tf = counter[word] / sum(counter.values())
        df = self._index[word].doc_freq
        idf = log(self._collection_length / df)
        return tf * idf

    #################################

    ######## QUERY-LENGTH VECTOR ########
    def _vectorise_query_length(self, query_words: list[str], passages: list[int]) -> tuple[dict[int, np.ndarray], np.ndarray]:
        query_length = len(query_words)

        passage_vectors = {}
        for pid in passages:
            passage_words = self._collection[pid]
            vector = np.zeros(len(passage_words))
            for idx, word in enumerate(passage_words):
                vector[idx] = self._index[word].get_posting(pid).tfidf
            passage_vectors[pid] = vector

            print(f"{query_words}")
            print(f"{self._collection[pid]}")
            print(f"{pid}: {vector}")
            print()

        query_vector = np.zeros(query_length)  # Maybe change to ones()?
        counter = Counter(query_words)
        for idx, word in enumerate(query_words):
            tf = counter[word] / query_length
            df = self._index[word].doc_freq
            idf = log(self._collection_length / df)
            query_vector[idx] = tf * idf
        print(query_vector)

        return passage_vectors, query_vector
    #################################

    ######## ORIGINAL VECTOR ########
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
            query_vectors[index] = tf * idf

        return query_vectors
    #################################

    @staticmethod
    def _similarity(query_vectors: np.ndarray, passage_vectors: np.ndarray) -> float:
        dot_product = np.dot(query_vectors, passage_vectors)
        norms = npl.norm(query_vectors) * npl.norm(passage_vectors)
        return dot_product / norms
