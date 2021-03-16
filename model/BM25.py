from math import log

from model.InvertedIndex import InvertedIndex
from model.util.TextProcessor import clean


class BM25:
    def __init__(self, index: 'InvertedIndex'):
        self._collection = index.collection
        self._index = index.index
        self._N = len(self._collection)
        self._k1 = 1.2
        self._b = 0.75
        self._avgdl = sum(float(len(x)) for x in self._collection.values()) / len(self._collection)

    def rank(self, query: str) -> dict[int, float]:
        scores = self.score_sums(query)
        scores = {key: value for key, value in
                  sorted(scores.items(), key=lambda item: item[1], reverse=True)[:100]}
        return scores

    def score_sums(self, query: str) -> dict[int, float]:
        relevant_docs = {}
        query_keywords = [word for word in clean(query)]
        for word in query_keywords:
            postings = self._index[word]
            for posting in postings:
                doc_id = posting.pointer
                if doc_id in relevant_docs:
                    relevant_docs[doc_id] += self.score(doc_id, word)
                else:
                    relevant_docs[doc_id] = self.score(doc_id, word)
        return relevant_docs

    def score(self, doc_id: int, word: str) -> float:
        postings = self._index[word]
        n = len(postings)
        idf = log(((self._N - n + 0.5) / (n + 0.5)) + 1)

        relevant_doc = next((posting for posting in postings if posting.pointer == doc_id), None)
        f = relevant_doc.freq
        dl = len(self._collection[doc_id])
        word_score = idf * ((f * (self._k1 + 1)) /
                            (f + self._k1 * (1 - self._b + self._b * (dl / self._avgdl))))
        return word_score
