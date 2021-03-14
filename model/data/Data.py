from dataclasses import dataclass


@dataclass
class Passage:
    pid: int
    text: str


@dataclass
class Query:
    qid: int
    query: str


@dataclass
class Posting:
    pointer: int
    freq: int
    positions: list[int]
    tfidf: int = 0
