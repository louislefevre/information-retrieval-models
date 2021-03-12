from model.data.Passage import Passage
from model.data.Word import Word
from model.data.Query import Query


# Text pre-processing
candidate_passages = "dataset/candidate_passages_top1000.tsv"
passages = Passage.convert(candidate_passages)

test_queries = "dataset/test-queries.tsv"
queries = Query.convert(test_queries)

passage_collection = "dataset/passage_collection_new.txt"
words = Word.convert(passage_collection)
