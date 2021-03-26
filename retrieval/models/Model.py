from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.util.TextProcessor import clean


class Model:
    def __init__(self, index: 'InvertedIndex'):
        self._collection = index.collection
        self._index = index.index

    def rank(self, query: str) -> dict[int, float]:
        query_tokens = self._clean_query(query)
        passages = self._relevant_passages(query_tokens)
        scores = self._score_query(query_tokens, passages)
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {key: value for key, value in scores}

    def _score_query(self, query_tokens: list[str], passages: list[int]) -> dict[int, float]:
        raise NotImplementedError

    def _clean_query(self, query: str):
        return [word for word in clean(query)]

    def _relevant_passages(self, query_words: list[str]) -> list[int]:
        return list(set(pid for word in query_words if word in self._index
                        for pid in self._index[word].postings))
