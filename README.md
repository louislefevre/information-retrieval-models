# Information Retrieval Models
Ranks passages against queries using various models and techniques.

## Models
- BM25 - Probabilistic retrieval model for estimating the relevance of a passage.
- VectorSpace - Algebraic model for representing passages as vectors.
- QueryLikelihood - Language model for calculating the likelihood of a document being relevant to a given query.

## How to Run
The program can be initialised by running *start.py*, which accepts parameters in the format of:  
`start.py <dataset> <model> [-s <smoothing>]`

### Parameters
#### Dataset
- The `<dataset>` parameter is required and is the path of the dataset to be parsed.
- Expects a TSV file in the format *<qid pid query passage>*, where qid is the query ID, pid is the ID of the passage retrieved, query is the query text, and passage is the passage text.
- Each column must be tab separated.

#### Model
- The `<model>` parameter is required and is the name of the model to be used for ranking passages.
- Expects either 'bm25' for the BM25 model, 'vs' for the Vector Space model, or 'lm' for the query likelihood model.
- Any other input will be deemed invalid, and an exception will be raised.

#### Smoothing
- The `-s <smoothing>` parameter is required only when using the Query Likelihood model, and is the name of the smoothing technique which will be applied.
- Expects either `laplace` for Laplace smoothing, `lidstone` for Lidstone smoothing, or `dirichlet` for Dirichlet smoothing.
- This parameter can only ever be used if the Query Likelihood model was selected for the `<model>` parameter, and an exception will be raised if any other model is used.

### Examples
- `start.py dataset.tsv bm25`
- `start.py dataset.tsv vs`
- `start.py dataset.tsv lm -s laplace`

## Dependencies
- [numpy](https://pypi.org/project/numpy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [nltk](https://pypi.org/project/nltk/)
- [num2words](https://pypi.org/project/num2words/)
- [tabulate](https://pypi.org/project/tabulate/)
- [punkt (nltk module)](http://www.nltk.org/api/nltk.tokenize.html?highlight=punkt)
- [stopwords (nltk module)](https://www.nltk.org/api/nltk.corpus.html)  
*NLTK modules are downloaded automatically at runtime*
