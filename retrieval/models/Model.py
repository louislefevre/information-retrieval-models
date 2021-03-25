from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.util.TextProcessor import clean


class Model:
    def __init__(self, index: 'InvertedIndex'):
        self._collection = index.collection
        self._index = index.index

    def rank(self, query: str) -> dict[int, float]:
        query_words = [word for word in clean(query)]
        scores = self._parse_query(query_words)
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {key: value for key, value in scores}

    def _parse_query(self, query: list[str]) -> dict[int, float]:
        raise NotImplementedError
