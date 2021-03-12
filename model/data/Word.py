from dataclasses import dataclass

from nltk import FreqDist

from model.util.FileReader import read
from model.util.TextProcessor import clean


@dataclass
class Word:
    word: str
    freq: float

    @staticmethod
    def convert(file_name: str) -> list:
        text = read(file_name)
        tokens = clean(text)
        dist = FreqDist(tokens)
        words = [Word(token, dist.freq(token)) for token in set(tokens)]
        words.sort(key=lambda word: word.freq, reverse=True)
        return words
