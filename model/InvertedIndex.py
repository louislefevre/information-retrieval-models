from model.data.Data import Posting, Passage
from model.util.TextProcessor import clean
from collections import Counter
from math import log


class InvertedIndex:
    def __init__(self, collection):
        self._collection = collection
        self._index = dict()

    def index(self):
        for passage in self._collection:
            self._index_passage(passage)

        for term, postings in self._index.items():
            for posting in postings:
                posting.tfidf = self._tf_idf(term)

    def _index_passage(self, passage):
        terms = clean(passage.text)

        seen_terms = dict()
        for pos, term in enumerate(terms):
            if term in seen_terms:
                term_frequency = seen_terms[term].freq
                positions = seen_terms[term].positions
            else:
                term_frequency = 0
                positions = []
            seen_terms[term] = Posting(passage.pid, term_frequency+1, positions+[pos])

        for term, posting in seen_terms.items():
            if term not in self._index:
                self._index[term] = [posting]
            else:
                self._index[term] += [posting]

    def _tf_idf(self, term: str, passage: Passage):
        # tf(t,d) = count of t in d / number of words in d
        for posting in self._index[term]:
            if posting.pointer == passage.pid:
                tf = posting.freq / len(passage.text.split())
                break
        else:
            raise KeyError(f"Term '{term}' is not in passage '{passage}.")

        # df(t) = occurrence of t in documents.
        df = len(self._index[term])

        # idf(t) = log(N/(df + 1))
        idf = log(len(self._collection) / df + 1)

        # tf-idf(t, d) = tf(t, d) * log(N/(df + 1))
        return tf * idf

    def display(self):
        for term, postings in self._index.items():
            posts = ""
            for post in postings:
                posts += str([post.pointer, post.freq, post.positions]) + " "
            print(f"{term} : {posts}")
