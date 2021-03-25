from dataclasses import dataclass
from retrieval.util.TextProcessor import clean_collection
from math import log


class InvertedIndex:
    def __init__(self, collection: dict[int, str]):
        self._collection = clean_collection(collection)
        self._index: dict[str, InvertedList] = dict()

    def parse(self):
        for pid, terms in self._collection.items():
            self._index_passage(pid, terms)
        for term, inverted_list in self._index.items():
            self._tfidf_passage(term, inverted_list)
        return self._index

    def _index_passage(self, pid: int, terms: list[str]):
        for pos, term in enumerate(terms):
            if term not in self._index:
                self._index[term] = InvertedList()
            self._index[term].add_posting(pid, pos)

    def _tfidf_passage(self, term: str, inv_index: 'InvertedList'):
        for pointer, posting in inv_index.postings.items():
            # Frequency of term in passage divided by number of words in the passage.
            tf = posting.freq / len(self._collection[pointer])
            # Number of occurrences of the term in all passages.
            df = self._index[term].doc_freq
            # Log total passage count divided by df.
            idf = log(len(self._collection) / df)
            posting.tfidf = tf * idf

    def display(self):
        for term, inv_index in self._index.items():
            posts = ""
            for pointer, posting in inv_index.postings.items():
                posts += str(f'{pointer}: {[posting.freq, posting.positions, posting.tfidf]}, ')
            print(f"{term} : {posts}")

    @property
    def collection(self) -> dict[int, list[str]]:
        return self._collection

    @property
    def index(self) -> dict[str, 'InvertedList']:
        return {key: value for key, value in sorted(self._index.items())}

    @property
    def collection_length(self) -> int:
        return len(self._collection)

    @property
    def avg_length(self) -> float:
        word_sum = sum(len(passage) for passage in self._collection.values())
        return word_sum / self.collection_length

    @property
    def vocab(self) -> list[str]:
        return list(self._index)

    @property
    def vocab_count(self) -> int:
        return len(self._index)


class InvertedList:
    def __init__(self):
        self._postings: dict[int, 'Posting'] = {}

    def add_posting(self, pid: int, position: int):
        if self.contains_posting(pid):
            return self.update_posting(pid, position)
        posting = Posting(1, [position])
        self._postings[pid] = posting

    def update_posting(self, pid: int, position: int):
        if not self.contains_posting(pid):
            return self.add_posting(pid, position)
        posting = self._postings[pid]
        posting.freq += 1
        posting.positions += [position]

    def get_posting(self, pid: int) -> 'Posting':
        return self._postings[pid]

    def contains_posting(self, pid: int) -> bool:
        return pid in self._postings

    @property
    def postings(self) -> dict[int, 'Posting']:
        return self._postings

    @property
    def doc_freq(self) -> int:
        return len(self._postings)


@dataclass
class Posting:
    freq: int
    positions: list[int]
    tfidf: float = 0.0
