from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.util.TextProcessor import clean


class Model:
    def __init__(self, index: 'InvertedIndex', mapping: dict[int, list[int]]):
        self._collection = index.collection
        self._index = index.index
        self._mapping = mapping

    def rank(self, qid: int, query: str) -> dict[int, float]:
        query_tokens = self._clean_query(query)
        passages = self._relevant_passages(qid, query_tokens)
        scores = self._score_query(query_tokens, passages)
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {key: value for key, value in scores}

    def _clean_query(self, query: str) -> list[str]:
        return [word for word in clean(query)]

    def _relevant_passages(self, qid: int, query_words: list[str]) -> list[int]:
        return list(set(pid for word in query_words if word in self._index
                        for pid in self._index[word].postings if pid in self._mapping[qid]))

    def _score_query(self, query_tokens: list[str], passages: list[int]) -> dict[int, float]:
        raise NotImplementedError
