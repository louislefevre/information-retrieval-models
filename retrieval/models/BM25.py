from math import log

from retrieval.InvertedIndex import InvertedIndex
from retrieval.util.TextProcessor import clean


class BM25:
    def __init__(self, index: 'InvertedIndex'):
        self._collection = index.collection
        self._index = index.index
        self._collection_length = len(self._collection)
        self._avg_length = sum(
            float(len(passage)) for passage in self._collection.values()) / len(self._collection)

    def rank(self, query: str, top_n=100) -> dict[int, float]:
        scores = self.score(query)
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
        return {key: value for key, value in scores}

    def score(self, query: str) -> dict[int, float]:
        passage_scores = {}
        query_keywords = [word for word in clean(query)]
        for word in query_keywords:
            if word not in self._index:
                continue
            postings = self._index[word]
            for posting in postings:
                pid = posting.pointer
                if pid in passage_scores:
                    passage_scores[pid] += self._score_passage(pid, word)
                else:
                    passage_scores[pid] = self._score_passage(pid, word)
        return passage_scores

    def _score_passage(self, pid: int, word: str) -> float:
        k1 = 1.2
        b = 0.75

        postings = self._index[word]
        passage_count = len(postings)
        idf = log(((self._collection_length - passage_count + 0.5) / (passage_count + 0.5)) + 1)

        relevant_doc = next((posting for posting in postings if posting.pointer == pid), None)
        freq = relevant_doc.freq
        passage_length = len(self._collection[pid])
        word_score = idf * ((freq * (k1 + 1)) /
                            (freq + k1 * ((1 - b) + b * (passage_length / self._avg_length))))
        return word_score
