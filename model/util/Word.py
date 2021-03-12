from model.util.FileReader import read
from model.TextProcessor import clean
from nltk import FreqDist


class Word:
    def __init__(self, word: str, freq: float):
        self._word = word
        self._freq = freq

    @property
    def word(self) -> str:
        return self._word

    @property
    def freq(self) -> float:
        return self._freq

    @staticmethod
    def convert(file_name: str) -> list[object]:
        text = read(file_name)
        tokens = clean(text)
        dist = FreqDist(tokens)
        return [Word(token, dist.freq(token)) for token in tokens]
