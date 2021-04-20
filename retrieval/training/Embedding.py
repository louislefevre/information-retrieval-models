import gensim.downloader as api
import numpy as np

from gensim.models.keyedvectors import KeyedVectors

from util.TextProcessor import clean_collection


def load_model(name: str) -> 'KeyedVectors':
    return api.load(name)


def mean_vector(model, words: list[str]) -> 'np.ndarray':
    words = [word for word in words if word in model.key_to_index]
    if not words:
        return np.empty((300,))
    return np.mean(model[words], axis=0)


def embed(data: dict[int, str]) -> dict[int, 'np.ndarray']:
    model = load_model('word2vec-google-news-300')
    data = clean_collection(data)
    return {id_: mean_vector(model, words) for id_, words in data.items()}
