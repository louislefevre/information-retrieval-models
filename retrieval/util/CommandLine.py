import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Information retrieval models.')

    # Dataset argument
    parser.add_argument('dataset', help='dataset for retrieving passages and queries')

    # Models arguments
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument('--bm25', action='store_true', help='BM25 model')
    models.add_argument('--vector', action='store_true', help='Vector Space model')
    models.add_argument('--query', action='store_true', help='Query Likelihood model')

    # Smoothing arguments
    smoothing = parser.add_mutually_exclusive_group(required=False)
    smoothing.add_argument('--laplace', action='store_true',
                           help='laplace smoothing for the Query Likelihood model')
    smoothing.add_argument('--lidstone', action='store_true',
                           help='lidstone smoothing for the Query Likelihood model')
    smoothing.add_argument('--dirichlet', action='store_true',
                           help='dirichlet smoothing for the Query Likelihood model')

    return parser.parse_args()
