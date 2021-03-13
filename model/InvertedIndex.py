from model.data.Data import Posting
from model.util.TextProcessor import clean


class InvertedIndex:
    def __init__(self):
        self._index = dict()

    def index_passage(self, passage):
        terms = clean(passage.passage)

        seen_terms = dict()
        for term in terms:
            if term in seen_terms:
                term_frequency = seen_terms[term].freq
            else:
                term_frequency = 0
            seen_terms[term] = Posting(passage.pid, term_frequency + 1)

        for term, posting in seen_terms.items():
            if term not in self._index:
                self._index[term] = [posting]
            else:
                self._index[term] += [posting]
