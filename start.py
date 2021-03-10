from model.util.DataManager import read, list_to_dict

queries = "dataset/test-queries.tsv"
candidate_passages = "dataset/candidate_passages_top1000.tsv"
passage_collection = "dataset/passage_collection_new.txt"

queries = read(queries)
candidate_passages = read(candidate_passages)
passage_collection = read(passage_collection)

queries = list_to_dict(queries, ['qid', 'query'])
candidate_passages = list_to_dict(candidate_passages, ['qid', 'pid', 'query', 'passage'])

for row in candidate_passages:
    print(row)
    break
