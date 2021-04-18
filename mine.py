import argparse


def main():
    parser = argparse.ArgumentParser(description='Information retrieval mining.')
    parser.add_argument('dataset', help='dataset with relevance information')
    parser.add_argument('results', help='results to evaluate')

    args = parser.parse_args()
    dataset = args.dataset
    results = args.results


if __name__ == '__main__':
    main()
