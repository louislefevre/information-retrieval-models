from retrieval.InvertedIndex import InvertedIndex, Posting
from retrieval.util.TextProcessor import clean


class VectorSpace:
    def __init__(self, index: 'InvertedIndex'):
        self._collection = index.collection
        self._index = index.index

    def rank(self, query: str, top_n=100) -> dict[int, float]:
        raise NotImplementedError

    def vectorise_query(self, query: str):
        query_vectors = {}
        query_keywords = [word for word in clean(query)]
        #for word in query_keywords:

        passage_vectors = self.vectorise_passages(query)

        for pid, vectors in passage_vectors.items():
            query_vectors = []
            for word in query_keywords:
                postings = self._index[word]
                # for posting in postings:

    def vectorise_passages(self, query: str) -> dict[int, list[float]]:
        passage_vectors = {}
        query_keywords = [word for word in clean(query)]
        for word in query_keywords:
            postings = self._index[word]
            for posting in postings:
                pid = posting.pointer
                if pid in passage_vectors:
                    vectorised_passage = passage_vectors[pid]
                else:
                    vectorised_passage = [0.0] * len(self._collection[pid])
                passage_vectors[pid] = self._vectorise_passage(vectorised_passage, posting)
        return passage_vectors

    @staticmethod
    def _vectorise_passage(vectorised_passage: list[float], posting: 'Posting') -> list[float]:
        for position in posting.positions:
            vectorised_passage[position] = posting.tfidf
        return vectorised_passage
