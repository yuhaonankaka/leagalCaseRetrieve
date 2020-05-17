This is the README file for A0195663R's and A0195919J's submission
Email: e0386266@u.nus.edu, e0386522@u.nus.edu


== Python Version ==

I'm (We're) using Python Version 3.6.6 for
this assignment.

== General Notes about this assignment ==

The assignment is divided into four parts:  Parsing, Query Expansion, Indexing, Searching.

Parsing:
    The Parser class will tokenize and then pre-process the query into tokens(terms). And if there is no """ and no "AND"
    in the query string, we can consider it as a Free Text Query. If any of the above condition is not satisfied, we can
    consider the query as a Boolean Text Query. Before we go further, we also need to check if the Boolean Text Query is
    in a correct format. Finally we can have a free text query like ["token1","token2",...] and a boolean text query
    like [["phrase_word1","phrase_word2"],["word3"],...]

    In the pre-processing of the tokens, we have tried excluding numbers and stopwords. But this results in a lower score.
    Thus we only converted the tokens into lower case and did stemming in the pre-processing step.


Query Expansion:
    1. Expansion using thesaurus
    For each tokens in the query, we will find its synonyms and append them to the original query.
    e.g. "quiet phone call" -> "quiet phone call silent tranquil calm telephone earphone shout ..."
    In this assignment, we used WordNet as the thesaurus source. Although WordNet synonyms contains many uncommon words,
    in our tf-idf scheme words that does not appear in the document will not contribute to the score. Thus it is
    acceptable to directly use WordNet synonyms.

    We also tried to implement a synonym selection scheme:
    We calculated the similarity between the synonym words and the original query word using WordNet's wup_similarity
    function, and only take the top K results to append to our query. However, this approach does not increase our
    evaluation score, so we decided not to use it in the final submission. The corresponding code can be found in
    query_parser.py, class Parser, function top_k_similarity and bestSimilarity.

    2. Pseudo relevance feedback
    We also tried to use pseudo relevance feedback as an experiment. In the list of documents returned by the original
    query, we extracted top 10 result documents and calculated the sum of document vectors. Because this operation
    needs the tf-idf score of each term in the document, we need to traverse the entire postings list thus it is extremely
    slow. In our experiment, calculating the sum of document vectors of the top 10 documents takes a minute. Besides,
    by inspecting the result document vector, we found that the terms that have the highest score are names,

    (e.g. ['rx', 'mamun', 'mosharaf', 'sobuj', 'koroitamana', 'eyl', 'shamim', 'leadbeat', 'hairgraph', 'bloodsworth'])

    because a name may have high term frequency in one legal document, but have very low document frequency in the whole
    collection. Observing these results, we decided not to use this form of pseudo relevance feedback in the submission.


Indexing:
    Index.py is built upon Assignment 3. In addition to the postings list and the term frequency information, we also
    added positional index in order to enable phrase query (we treated them as proximity queries). We saved the index
    in binary form, where each entry (a term) takes the format:
        [length of entry (number of items), term_id, length of postings (integer count),
        postings: docID1, docID2, ..., tf1, tf2 ...,
        positions: length1, pos1_1, pos1_2, ..., length2, pos2_1, ...]
     The dictionary is stored to file as binary using pickle.


Searching:
    Searching feature is built upon Assignment 3. Besides the basic tf-idf ranking (lnc.ltc), we added a proximity score
    for each query by using the positional index. The proximity score of each document is calculated as following:
        We calculate a distance matrix for each pair of tokens in the query (e.g. "t1 t2 t3 t4 t5"):
                                                t1  t2  t3  t4  t5
                                            t1  --  d1  d2  d3  d4
                                            t2  --  --  d5  d6  d7
                                            t3  --  --  --  d8  d9
                                            t4  --  --  --  --  d10
                                            t5  --  --  --  --  --
        the distance d1, d2, ... d10 is calculated by:
            for each token pair (w1, w2), we get their positions in the document:
                w1: [p1_1, p1_2, p1_3 ...]
                w2: [p2_1, p2_2, p2_3 ...]
            then the distance between w1 and w2 is given by:
                distance = min { abs( p1_i - p2_j) for each i,j }
            (the minimum distance between any occurrence of word w1 and word w2 in document)
        Then, the final proximity score of the document is given by:
            W_prox = sum {1/d_i, for all i} / (n*(n-1)/2), where n is the length of the query
        (sum of the inverse of distance, normalized by the maximum possible number of pairs)
        Note that if a query token is absent in the document, we ignore its corresponding distance in the calculation of
        proximity score (but we still normalize the score using the number of token pairs in the query). This means a
        document that do not contain some query tokens will be penalized comparing to those contain them.

    The final score of a document is a weighted sum of the cosine similarity and the proximity score:
        W_final = alpha * W_cosine + (1 - alpha) * W_prox

    We process free text queries directly using the scheme mentioned above, setting the value of alpha to 0.8 (emphasize
    cosine similarity).

    When dealing with Boolean queries, we first process each sub queries using the above scheme, setting the value of
    alpha to 0.2 (emphasize proximity score to simulate phrase query). Then, for each document, we sum the score of each
    sub query to obtain the final score. In this way, we are able to return documents that do not strictly match the query
    while letting better matched documents rank higher.

    We have also tried other ways to handle boolean queries, e.g. only return the documents that match every sub query,
    but these experiments resulted in lower score compared to the scheme mentioned above.


== Files included with this submission ==

dictionary.txt
Description: file containing the dictionary (binary file saved using pickle)

postings.txt
Description: file containing the calculated postings (binary file)

README.txt
Description: this README file

query.py
Description: containing the Query class, have 3 different types and query string can be stored in it

dictionary.py
Description: containing the Dictionary class.

query_parser.py
Description: containing the Parser class, this will be used to pre-process the queries.

index.py
Description: main file used for constructing index. Containing the Indexer class.

search.py
Description: main file used to process queries and retrieve answers. Containing the SearchEngine class.

/nltk_data
Description: stop word data used by nltk.stopwords

== Statement of individual work ==

Please initial one of the following statements.

[X] I, A0195663R, A0195919J, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

I suggest that I should be graded as follows:

<Please fill in>

== References ==

https://docs.python.org/3.6/library/heapq.html
The usage of python heapq

https://pythonprogramming.net/wordnet-nltk-tutorial/
The usage of wordNet

Princeton University "About WordNet." WordNet. Princeton University. 2010.
https://wordnet.princeton.edu/
WordNet online version

https://stackoverflow.com/questions/24989772/finding-closest-numbers-within-two-arrays
An linear time algorithm to find the closets numbers in two sorted arrays