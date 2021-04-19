from math import log2

from numpy import mean


def avg_precision(ranks: list[int], relevant: list[int]) -> float:
    rel_counter = 0
    documents = {}
    for i, doc in enumerate(ranks, start=1):
        if doc in relevant:
            rel_counter += 1
        documents[doc] = (rel_counter / i)

    total_precision = sum([documents[doc] for doc in relevant if doc in documents])
    return total_precision / len(relevant)


def mean_average_precision(results: dict[int, dict[int, float]], relevant: dict[int, list[int]]) -> float:
    all_precision = []
    for qid, ranks in results.items():
        all_precision.append(avg_precision(list(ranks), relevant[qid]))
    return mean(all_precision)


def ndcg(ranks: list[int], relevant: list[int]) -> float:
    dcg_total = 0.0
    for pos, pid in enumerate(ranks, start=1):
        rel = 1.0 if pid in relevant else 0.0
        dcg = rel / log2(pos + 1)
        dcg_total += dcg

    idcg_total = 0.0
    for pid in ranks:
        if pid in relevant:
            ranks.insert(0, ranks.pop(ranks.index(pid)))
    for pos, pid in enumerate(ranks, start=1):
        rel = 1.0 if pid in relevant else 0.0
        dcg = rel / log2(pos + 1)
        idcg_total += dcg

    if idcg_total == 0.0:
        return 0.0

    return dcg_total / idcg_total


def mean_ndcg(results: dict[int, dict[int, float]], relevant: dict[int, list[int]]) -> float:
    all_ndcg = []
    for qid, ranks in results.items():
        all_ndcg.append(ndcg(list(ranks), relevant[qid]))
    return mean(all_ndcg)
