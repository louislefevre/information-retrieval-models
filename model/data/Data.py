from dataclasses import dataclass


@dataclass
class Passage:
    pid: int
    passage: str


@dataclass
class Query:
    qid: int
    query: str


@dataclass
class Posting:
    pointer: int
    freq: int
