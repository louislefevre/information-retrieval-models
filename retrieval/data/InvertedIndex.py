import itertools
from collections import Counter
from dataclasses import dataclass

from retrieval.util.Math import tf_idf
from retrieval.util.TextProcessor import clean_collection


class InvertedIndex:
    def __init__(self, collection: dict[int, str]):
        self._collection = clean_collection(collection)
        self._index: dict[str, InvertedList] = dict()

    def parse(self):
        self._index_passages()
        self._tfidf_passages()

    def _index_passages(self):
        for pid, passage in self._collection.items():
            for term in passage:
                if term not in self._index:
                    self._index[term] = InvertedList()
                self._index[term].add_posting(pid)

    def _tfidf_passages(self):
        for term, inv_index in self._index.items():
            for pointer, posting in inv_index.postings.items():
                posting.tfidf = tf_idf(posting.freq, self._index[term].doc_freq, len(self._collection))

    @property
    def index(self) -> dict[str, 'InvertedList']:
        return {key: value for key, value in sorted(self._index.items())}

    @property
    def collection(self) -> dict[int, list[str]]:
        return self._collection

    @property
    def words(self) -> list[str]:
        return list(itertools.chain.from_iterable(self._collection.values()))

    @property
    def vocab(self) -> list[str]:
        return sorted(self._index)

    @property
    def collection_length(self) -> int:
        return len(self._collection)

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def vocab_count(self) -> int:
        return len(self._index)

    @property
    def avg_length(self) -> float:
        return self.word_count / self.collection_length

    @property
    def counter(self) -> Counter[str]:
        return Counter(self.words)


class InvertedList:
    def __init__(self):
        self._postings: dict[int, 'Posting'] = dict()

    def add_posting(self, pid: int):
        if self.contains_posting(pid):
            return self.update_posting(pid)
        posting = Posting()
        self._postings[pid] = posting

    def update_posting(self, pid: int):
        if not self.contains_posting(pid):
            return self.add_posting(pid)
        posting = self._postings[pid]
        posting.freq += 1

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
    freq: int = 1
    tfidf: float = 0.0
