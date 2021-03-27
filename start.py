from retrieval.QueryParser import QueryParser
from retrieval.util.CommandLine import parse_args


def main():
    args = parse_args()
    parser = QueryParser(args.dataset)
    model = None
    smoothing = None

    if args.bm25:
        model = 'bm25'
    elif args.vector_space:
        model = 'vector'
    elif args.query_likelihood:
        model = 'query'
        if args.laplace:
            smoothing = 'laplace'
        elif args.lidstone:
            smoothing = 'lidstone'
        elif args.dirichlet:
            smoothing = 'dirichlet'

    results = parser.parse(model, smoothing=smoothing)


if __name__ == '__main__':
    main()
