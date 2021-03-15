# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from num2words import num2words


def clean_collection(collection: dict[int, str]) -> dict[int, list[str]]:
    return {pid: clean(passage) for pid, passage in collection.items()}


def clean(text: str, remove_sw=True) -> list[str]:
    tokens = _tokenize(text)
    tokens = _convert_numbers(tokens)
    tokens = _normalise(tokens)
    if remove_sw:
        tokens = _remove_stopwords(tokens)
    tokens = _stem(tokens)
    return tokens


def _tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def _convert_numbers(tokens: list[str]) -> list[str]:
    # isascii() is required for handling unicode fractions (e.g. '½').
    return [num2words(word) if word.isnumeric() and word.isascii() else word for word in tokens]


def _normalise(tokens: list[str]) -> list[str]:
    return [word.lower() for word in tokens if word.isalpha()]


def _remove_stopwords(tokens: list[str]) -> list[str]:
    stop_words = stopwords.words("english")
    return [word for word in tokens if word not in stop_words]


def _stem(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]
