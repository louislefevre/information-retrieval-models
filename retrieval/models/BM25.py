from math import log

from retrieval.InvertedIndex import InvertedIndex
from retrieval.models.Model import Model


class BM25(Model):
    def __init__(self, index: 'InvertedIndex'):
        super().__init__(index)
        self._collection_length = index.collection_length
        self._avg_length = index.avg_length

    def _score_passage(self, pid: int, word: str) -> float:
        k1 = 1.2
        b = 0.75

        inv_index = self._index[word]
        postings = inv_index.postings
        passage_count = inv_index.doc_freq
        idf = log(((self._collection_length - passage_count + 0.5) / (passage_count + 0.5)) + 1)

        relevant_doc = next((posting for posting in postings if posting.pointer == pid), None)
        freq = relevant_doc.freq
        passage_length = len(self._collection[pid])
        word_score = idf * ((freq * (k1 + 1)) /
                            (freq + k1 * ((1 - b) + b * (passage_length / self._avg_length))))
        return word_score
