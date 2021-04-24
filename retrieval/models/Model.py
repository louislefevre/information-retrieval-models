from inverted_index import InvertedIndex
from retrieval.util.TextProcessor import clean


class Model:
    def __init__(self, index: 'InvertedIndex', mapping: dict[str, list[str]]):
        self._index = index
        self._documents = index.documents
        self._mapping = mapping

    def rank(self, qid: str, query: str) -> dict[str, float]:
        query_words = self._clean_query(query)
        passages = self._relevant_passages(qid, query_words)

        scores = {}
        for pid in passages:
            scores[pid] = self._score_passage(pid, query_words)

        ranks = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {pid: score for pid, score in ranks}

    def _clean_query(self, query: str) -> list[str]:
        return [word for word in clean(query)]

    def _relevant_passages(self, qid: str, query_words: list[str]) -> list[str]:
        return list(set(pid for word in query_words if word in self._index
                        for pid in self._index.postings(word) if pid in self._mapping[qid]))

    def _score_passage(self, pid: str, query_words: list[str]) -> float:
        raise NotImplementedError
