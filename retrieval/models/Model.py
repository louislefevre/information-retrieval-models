from retrieval.data.InvertedIndex import InvertedIndex
from retrieval.util.TextProcessor import clean


class Model:
    def __init__(self, index: 'InvertedIndex', mapping: dict[int, list[int]]):
        self._collection = index.collection
        self._index = index.index
        self._mapping = mapping

    def rank(self, qid: int, query: str) -> dict[int, float]:
        query_words = self._clean_query(query)
        passages = self._relevant_passages(qid, query_words)

        scores = {}
        for idx, pid in enumerate(passages):
            scores[pid] = self._score_passage(pid, query_words)
            print(f"{idx}/{len(passages)} passages parsed for query {qid}      ", end='\r')

        ranks = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {pid: score for pid, score in ranks}

    def _clean_query(self, query: str) -> list[str]:
        return [word for word in clean(query)]

    def _relevant_passages(self, qid: int, query_words: list[str]) -> list[int]:
        return list(set(pid for word in query_words if word in self._index
                        for pid in self._index[word].postings if pid in self._mapping[qid]))

    def _score_passage(self, pid: int, query_words: list[str]) -> float:
        raise NotImplementedError
