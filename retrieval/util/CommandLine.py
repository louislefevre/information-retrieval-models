import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Information retrieval models.')

    # Dataset argument
    parser.add_argument('dataset', dest='dataset',
                        help='dataset for retrieving passages and queries')

    # Models arguments
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument('--bm25', dest='bm25', help='BM25 model')
    models.add_argument('--vector', dest='vector_space', help='Vector Space model')
    models.add_argument('--query', dest='query_likelihood', help='Query Likelihood model')

    # Smoothing arguments
    smoothing = parser.add_mutually_exclusive_group(required=False)
    smoothing.add_argument('--laplace', dest='laplace',
                           help='laplace smoothing for the Query Likelihood model')
    smoothing.add_argument('--lidstone', dest='lidstone',
                           help='lidstone smoothing for the Query Likelihood model')
    smoothing.add_argument('--dirichlet', dest='dirichlet',
                           help='dirichlet smoothing for the Query Likelihood model')

    return parser.parse_args()
