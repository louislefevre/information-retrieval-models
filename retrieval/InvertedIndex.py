from dataclasses import dataclass
from retrieval.util.TextProcessor import clean_collection
from math import log10


class InvertedIndex:
    def __init__(self, collection: dict[int, str]):
        self._collection = clean_collection(collection)
        self._index: dict[str, list['Posting']] = dict()

    def index_collection(self):
        for pid, terms in self._collection.items():
            self._index_passage(pid, terms)

        for term, postings in self._index.items():
            for posting in postings:
                posting.tfidf = self._tf_idf(term, posting)
        return self._index

    def _index_passage(self, pid: int, terms: list[str]):
        seen_terms = dict()
        for pos, term in enumerate(terms):
            if term in seen_terms:
                term_frequency = seen_terms[term].freq
                positions = seen_terms[term].positions
            else:
                term_frequency = 0
                positions = []
            seen_terms[term] = Posting(pid, term_frequency+1, positions+[pos])

        for term, posting in seen_terms.items():
            if term not in self._index:
                self._index[term] = [posting]
            else:
                self._index[term] += [posting]

    def _tf_idf(self, term: str, posting: 'Posting') -> float:
        # Frequency of term in passage divided by number of words in the passage
        tf = posting.freq / len(self._collection[posting.pointer])

        # Number of occurrences of the term in all passages.
        df = len(self._index[term])

        # Log total passage count divided by df.
        idf = log10(len(self._collection) / df)

        return tf * idf

    def display(self):
        for term, postings in self._index.items():
            posts = ""
            for post in postings:
                posts += str([post.pointer, post.freq, post.tfidf, post.positions]) + " "
            print(f"{term} : {posts}")

    @property
    def collection(self) -> dict[int, list[str]]:
        return self._collection

    @property
    def index(self) -> dict[str, list['Posting']]:
        return self._index

    @property
    def collection_length(self) -> int:
        return len(self._collection)

    @property
    def avg_length(self) -> float:
        word_sum = sum(len(passage) for passage in self._collection.values())
        return word_sum / self.collection_length


@dataclass
class Posting:
    pointer: int
    freq: int
    positions: list[int]
    tfidf: float = 0.0
