from dataclasses import dataclass

from nltk import FreqDist

from model.util.FileReader import read
from model.util.TextProcessor import clean


@dataclass
class Passage:
    pid: int
    query: str

    @staticmethod
    def convert(file_name: str) -> list:
        rows = read(file_name)
        return [Passage(int(row[0]), row[1]) for row in rows]


@dataclass
class Query:
    qid: int
    query: str

    @staticmethod
    def convert(file_name: str) -> list:
        rows = read(file_name)
        return [Query(int(row[0]), row[1]) for row in rows]


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
