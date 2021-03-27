from math import log

import numpy as np
import numpy.linalg as npl


def normalise(data: iter) -> list[float]:
    min_, max_ = min(data), max(data)
    return [(i - min_) / (max_ - min_) for i in data]


def tf_idf(tf: int, dl: int, df: int, n: int) -> float:
    tf = tf / dl
    idf = log(n / df)
    return tf * idf


def cos_sim(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    dot_product = np.dot(vector_1, vector_2)
    norms = npl.norm(vector_1) * npl.norm(vector_2)
    return dot_product / norms
