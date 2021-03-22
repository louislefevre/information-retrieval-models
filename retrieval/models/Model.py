from retrieval.InvertedIndex import InvertedIndex
from retrieval.util.TextProcessor import clean


class Model:
    def __init__(self, index: 'InvertedIndex'):
        self._collection = index.collection
        self._index = index.index

    def rank(self, query: str, top_n=100) -> dict[int, float]:
        scores = self._parse_query(query)
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
        return {key: value for key, value in scores}

    def _parse_query(self, query: str) -> dict[int, float]:
        passage_scores = {}
        query_keywords = [word for word in clean(query)]
        for word in query_keywords:
            if word not in self._index:
                continue
            inv_list = self._index[word]
            for pid, posting in inv_list.postings.items():
                if pid in passage_scores:
                    passage_scores[pid] += self._score_passage(pid, word)
                else:
                    passage_scores[pid] = self._score_passage(pid, word)
        return passage_scores

    def _score_passage(self, pid: int, word: str) -> float:
        raise NotImplementedError
