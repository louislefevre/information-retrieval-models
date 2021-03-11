from model.util.DataManager import read, lists_to_dicts
from model.TextProcessor import tokenize, normalise, remove_stopwords, stem

# queries = "dataset/test-queries.tsv"
# candidate_passages = "dataset/candidate_passages_top1000.tsv"

# queries = read(queries)
# candidate_passages = read(candidate_passages)
# passage_collection = read(passage_collection)

# queries = lists_to_dicts(queries, ['qid', 'query'])
# candidate_passages = lists_to_dicts(candidate_passages, ['qid', 'pid', 'query', 'passage'])

# statistics = TextProcessor(passage_collection)
# frequencies = statistics.frequency()

passage_collection = "dataset/passage_collection_new.txt"
tokens = tokenize(passage_collection)
tokens = normalise(tokens)
tokens = remove_stopwords(tokens)
tokens = stem(tokens)
