# Information Retrieval Models
Ranks passages against queries using various models and techniques.

## Structure
The following describes the purpose of each package and the contained modules.
### Retrieval
- data/ - Data structures for storing and managing data.
- models/ - Information retrieval models for ranking passages against queries.
- util/ - Helper functions used for processing and managing data.
- DatasetParser.py - The primary module which takes a dataset as input and parses it using a specified model.
### Data
- Dataset.py - Extracts and retrieves data from the dataset.
- InvertedIndex.py - Indexes passages from the dataset.
### Models
- Model.py - Base class which all IR models inherit.
- BM25.py - BM25 probabilistic retrieval model for estimating the relevance of a passage.
- VectorSpace.py - Vector space algebraic model for representing passages as vectors.
- QueryLikelihood.py - Query likelihood language model for calculating the likelihood of a document being relevant to a 
given query.
### Util
- FileManager.py - Reads and writes to a given file.
- TextProcessor.py - Performs text preprocessing on a collection or passage.
- Plotter.py - Plots the term frequency graph.
- Math.py - Various mathematical formula functions.

## How to Run
The program can be initialised by running *start.py*, which accepts parameters in the format of:
`start.py <dataset> <model> [-s <smoothing>] [-p]`
### Parameters
#### Dataset
- The `<dataset>` parameter is required and is the path of the dataset to be parsed.
- Expects a TSV file in the format *<qid pid query passage>*, where qid is the query ID, pid is the 
  ID of the passage retrieved, query is the query text, and passage is the passage text, 
- Each column must be tab separated.
#### Model
- The `<model>` parameter is required and is the name of the model to be used for ranking passages.
- Expects either 'bm25' for the BM25 model, 'vs' for the Vector Space model, or 'lm' for the query 
  likelihood model.
- Any other input will be deemed invalid, and an exception will be raised.
#### Smoothing
- The `-s <smoothing>` parameter is required only when using the Query Likelihood model, and is 
  the name of the smoothing technique which will be applied.
- Expects either 'laplace' for Laplace smoothing, 'lidstone' for Lidstone smoothing, or 'dirichlet' 
  for Dirichlet smoothing.
- This parameter can only ever be used if the Query Likelihood model was selected for the `<model>` 
  parameter, and an exception will be raised if any other model is used.
#### Frequency Plot
- The `-p` parameter is optional and generates a PNG file which displays term frequencies in graph 
  format.
- By default, this file will be saved to the local directory as *term-frequencies.png*.

### Examples
#### BM25 Model
`start.py dataset/candidate_passages_top1000.tsv bm25`
#### Vector Space Model
`start.py dataset/candidate_passages_top1000.tsv vs`
#### Query Likehood Model
`start.py dataset/candidate_passages_top1000.tsv lm -s laplace`  
`start.py dataset/candidate_passages_top1000.tsv lm -s lidstone`  
`start.py dataset/candidate_passages_top1000.tsv lm -s dirichlet`  

## Dependencies
- [numpy](https://pypi.org/project/numpy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [nltk](https://pypi.org/project/nltk/)
- [num2words](https://pypi.org/project/num2words/)
- [punkt (nltk module)](http://www.nltk.org/api/nltk.tokenize.html?highlight=punkt)
- [stopwords (nltk module)](https://www.nltk.org/api/nltk.corpus.html)  
*NLTK modules are downloaded automatically at runtime*
