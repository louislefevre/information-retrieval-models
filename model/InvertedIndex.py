from model.data.Data import Posting
from model.util.TextProcessor import clean


class InvertedIndex:
    def __init__(self):
        self._index = dict()

    def index_passage(self, passage):
        terms = clean(passage.passage)
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

    def display(self):
        for term, postings in self._index.items():
            posts = ""
            for post in postings:
                posts += str([post.pointer, post.freq, post.positions]) + " "
            print(f"{term} : {posts}")
