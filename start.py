import argparse

from retrieval.QueryParser import QueryParser


def main():
    parser = argparse.ArgumentParser(description='Information retrieval models.')
    parser.add_argument('dataset', help='dataset for retrieving passages and queries')
    parser.add_argument('model', help='model for ranking passages against queries')
    parser.add_argument('-s', '--smoothing', help='smoothing for the Query Likelihood model')

    args = parser.parse_args()
    model = args.model
    smoothing = args.smoothing

    parser = QueryParser(args.dataset)
    results = parser.parse(model, smoothing=smoothing)


if __name__ == '__main__':
    main()
