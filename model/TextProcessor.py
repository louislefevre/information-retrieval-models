# nltk.download('punkt')
# nltk.download('stopwords')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def clean(file):
    tokens = tokenize(file)
    tokens = normalise(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return tokens


def tokenize(file):
    text = open(file, 'r').read()[:100000]  # Temporary for testing
    return word_tokenize(text)


def normalise(tokens):
    return [word.lower() for word in tokens if word.isalpha()]


def remove_stopwords(tokens):
    stop_words = stopwords.words("english")
    return [word for word in tokens if word not in stop_words]


def stem(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]
