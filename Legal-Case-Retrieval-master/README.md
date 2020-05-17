# Legal Case Retrieval Project Design
## Preprocessing & Parsing
### Features
* Query Parsing
  - [x] Free text query
  - [x] Boolean query + phrase query
* Preprocessing
  * [x] Tokenization
  * [x] Stopword Removal
  * [ ] ~~Deal with numbers~~
### Interfaces
In parser.py:
*  method **parse_query**:
  * Input: query string
  * Output: 
    * Free text query: A Query object with type "FreeText"
    * Boolean + Phrase query: A Query object with type "Boolean"
* method **preprocess**:
  * Input: String to preprocess
  * Output: A list of preprocessed tokens
## Query
### Features
- [x] Provides container for queries
### Interfaces
In query.py:
* class Query:
  * type: "FreeText" or "Boolean"
  * Data:
    * if "FreeText": a list of tokens in the free text query
    * if "Boolean": a list of list of tokens
      e.g. `"fertility treatment" AND damages`  -> `[["fertility", "treatment"], ["damages"]]`
## Index
### Features
- [x] Read input file
- [x] Construct a dictionary
- [x] TF-IDF index + vector space model (for free text queries)
- [x] Save the dictionary and index to file
* [x] Positional index (for phrase queries)
* [ ] *~~Topic based ranking~~*
### Interfaces
In index.py
* main function with arguments specified by HW4 website
* class Indexer:
  * method **index**: creates the main index
    * Inputs: path to the file to be indexed
    * Outputs: None
  * method **save**: save index & dictionary to file
    * Inputs: path to the output index file and dictionary file
    * Outputs: None
* class Dictionary:
  * list **itos**: mapping termID -> token
  * dict **stoi**: mapping token -> termID
  * dict **dfs**: mapping token -> document frequency
  * method **save**: saves the dictionary to file
    * Inputs: path to the dictionary file
    * Outputs: None
  * ...
## Search
### Features
- [x] Cosine similarity ranking
- [x] Proximity weighting
- [x] *Query Refinement*
  * [x] Query Expansion
  * [ ] ~~Relevance Feedback~~
### Interface
* class SearchEngine:
  * constructor:
    * Inputs: path to dictionary file, path to postings file
  * method query:
    * Inputs: a query string
    * Outputs: list of relevant documents

# Milestones
## Before Apr 13 (Sat)
- [x] Finish query parsing and preprocessing
- [x] Finish read input file
- [x] Finish query container
- [x] Be able to handle boolean queries (without proximity search)

After this we will have a minimally working version.
## Before  Apr 18 (Thu)
- [x] Proximity search support
- [ ] ~~Index with zones / fields~~
- [x] Query expansion using thesaurus (WordNet should be useful)

After this the project is ready for submission (without bonus)

## Before Apr 21 (Sun), Submission DDL
- [x] Explore second query expansion (Bonus)
- [ ] ~~Explore topic based ranking (Bonus)~~