import argparse

from retrieval.DatasetParser import DatasetParser


def main():
    parser = argparse.ArgumentParser(description='Information retrieval models.')
    parser.add_argument('dataset', help='dataset for retrieving passages and queries')
    parser.add_argument('model', help='model for ranking passages against queries')
    parser.add_argument('-p', '--plot', action='store_true', help='generate term frequency graph')
    parser.add_argument('-s', '--smoothing', help='smoothing for the Query Likelihood model')

    args = parser.parse_args()
    parser = DatasetParser(args.dataset)
    results = parser.parse(args.model, plot_freq=args.plot, smoothing=args.smoothing)


if __name__ == '__main__':
    main()
