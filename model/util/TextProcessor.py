# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from num2words import num2words


def clean(text: str) -> list[str]:
    tokens = tokenize(text)
    tokens = convert_numbers(tokens)
    tokens = normalise(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return tokens


def tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def convert_numbers(tokens: list[str]) -> list[str]:
    return [num2words(word) if word.isnumeric() else word for word in tokens]


def normalise(tokens: list[str]) -> list[str]:
    return [word.lower() for word in tokens if word.isalpha()]


def remove_stopwords(tokens: list[str]) -> list[str]:
    stop_words = stopwords.words("english")
    return [word for word in tokens if word not in stop_words]


def stem(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]
