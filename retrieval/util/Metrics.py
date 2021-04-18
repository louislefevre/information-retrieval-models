from numpy import mean


def avg_precision(ranks: list[int], relevant: list[int]):
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
