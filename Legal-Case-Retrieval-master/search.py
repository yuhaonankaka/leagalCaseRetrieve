#!/usr/bin/python
import math
import heapq
import sys
import getopt
import pickle
import os
import bisect
import nltk
nltk.data.path.append('./nltk_data')
from nltk.corpus import wordnet as wn
from query_parser import Parser


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")


class Postings(object):
    def __init__(self, postings=[], skip_intv=0, skip_ptrs=[]):
        self.postings = postings
        self.skip_intv = skip_intv
        self.skip_ptrs = skip_ptrs


class SearchEngine(object):
    def __init__(self, dict_file, post_file, encoding_length=3):
        self.dictionary = self.__load(dict_file)
        self.postings_file = open(post_file, "rb")
        self.parser = Parser()
        self.encoding_length = encoding_length

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.postings_file.close()

    def __load(self, dict_file):
        """
        Loads the dictionary from file
        :param dict_file: the file containing the dictionary
        :return: the dictionary object
        """
        with open(dict_file, "rb") as f:
            dictionary = pickle.load(f)
        return dictionary

    def __get_synonyms(self, token):
        result = set()
        for synset in wn.synsets(token):
            name = self.parser.preprocess([synset.name().split(".")[0]])[0]
            if name in self.dictionary.stoi:
                result.add(name)
        # if len(result) > 5:
        #     result_list = [word for word, sim in self.parser.top_k_similarity(token, list(result), 5)]
        #     result = set(result_list)
        return result

    def __expand_query(self, query):
        synonyms = set()
        for token in query:
            synonyms = synonyms.union(set(self.__get_synonyms(token)))
        return query + list(synonyms)

    def __parse_byte_repr(self, byte_repr, to_int):
        """
        Parses byte representation of a postings list, ordered as:
            [termID, length of postings (integer count)
            postings: docID1, docID2, ..., tf1, tf2 ...,
            positions: length1, pos1_1, pos1_2, ..., length2, pos2_1, ...]
        :param byte_repr: byte representation to parse
        :param to_int: the function used to convert byte to integer
        :return: a Postings object
        """
        # read length of postings
        postings_length = to_int(byte_repr[self.encoding_length:self.encoding_length*2])
        # read postings
        postings = []
        idx = self.encoding_length*2
        for i in range(postings_length):
            postings.append(to_int(byte_repr[idx:idx+self.encoding_length]))
            idx += self.encoding_length
        # read tf values
        tfs = []
        for i in range(postings_length):
            tfs.append(to_int(byte_repr[idx:idx+self.encoding_length]))
            idx += self.encoding_length
        # read positions
        positions = [[] for _ in range(postings_length)]
        for i in range(postings_length):
            pos_length = to_int(byte_repr[idx:idx + self.encoding_length])
            idx += self.encoding_length
            for _ in range(pos_length):
                positions[i].append(to_int(byte_repr[idx:idx + self.encoding_length]))
                idx += self.encoding_length
        return postings, tfs, positions

    def __get_postings(self, token):
        """
        Looks up a postings list in file.
        :param token: the token to loop up
        :return: a tuple of (termID, term frequency, postings)
        """
        def to_int(x): return int.from_bytes(x, byteorder="little")
        fp = self.dictionary[token]
        if fp is not None:
            # if token exists in dictionary
            self.postings_file.seek(fp)
            # read length of entry
            entry_len = to_int(self.postings_file.read(self.encoding_length))
            remain_entry = self.postings_file.read((entry_len - 1)*self.encoding_length)
            # read postings
            postings, tfs, positions = self.__parse_byte_repr(remain_entry, to_int)
            return postings, tfs, positions
        else:
            return None

    def __get_query_tfs(self, tokens):
        """
        Gets the term frequencies from a query (a series of tokens)
        :param tokens: the token series of the query
        :return: a tuple of (sorted_tokens, tfs), where sorted_tokens is a list of terms that appear in the query (with
            no repetition and in alphabetical order) and tf is the corresponding term frequencies of the terms.
        """
        token_dict = {}
        tfs = []
        for token in tokens:
            if token not in self.dictionary.stoi:
                continue
            if token not in token_dict:
                token_dict[token] = 1
            else:
                token_dict[token] += 1
        sorted_tokens = sorted(token_dict.keys())
        for token in sorted_tokens:
            tfs.append(token_dict[token])
        return sorted_tokens, tfs

    def __get_idf(self, token):
        """
        Returns the inverse document frequency for a term
        :param token: the term to look up
        :return: the inverse document frequency
        """
        df = self.dictionary.dfs.get(token, 0)
        if df == 0:
            return 0
        else:
            return math.log(len(self.dictionary.doc_ids)/df, 10)

    # algorithm from https://stackoverflow.com/questions/24989772/finding-closest-numbers-within-two-arrays
    def __min_dist(self, l1, l2):
        i = 0
        j = 0
        min_val = abs(l1[0] - l2[0])
        while i < len(l1) and j < len(l2):
            min_val = min(min_val, abs(l1[i] - l2[j]))
            if i+1 == len(l1) or j+1 == len(l2):
                break
            if abs(l1[i+1] - l2[j]) < abs(l1[i] - l2[j+1]):
                i += 1
            else:
                j += 1
        if i == len(l1) -1:
            while j < len(l2):
                min_val = min(min_val, abs(l1[i] - l2[j]))
                j += 1
        else:
            while i < len(l1):
                min_val = min(min_val, abs(l1[i] - l2[j]))
                i += 1
        return min_val

    def __search_similarity(self, query_tokens, query_tfidfs, alpha=0.8):
        """
        Search similar documents for query.
        :param query_tokens: the terms in the query
        :param query_tfidfs: corresponding term tfidfs (in the same order with query_tokens)
        :return: a list of document ids, ranked by similarity
        """
        heap = [(0, doc_id) for doc_id in self.dictionary.doc_ids]
        position = [[] for _ in self.dictionary.doc_ids]
        for query_idx, token in enumerate(query_tokens):
            # get postings
            postings, tfs, positions = self.__get_postings(token)
            for doc_idx, doc_id in enumerate(postings):
                position[self.dictionary.doc_order[doc_id]].append(positions[doc_idx])
                similarity, doc_id = heap[self.dictionary.doc_order[doc_id]]
                similarity -= (1 + math.log(tfs[doc_idx], 10))/self.dictionary.doc_len[doc_id] * query_tfidfs[query_idx]
                heap[self.dictionary.doc_order[doc_id]] = (similarity, doc_id)
        # calculate proximity score for each document
        for doc_order, pos in enumerate(position):
            n_hit = len(pos)
            if n_hit > 1:
                min_dist = [1 / self.__min_dist(pos[i], pos[j]) for i in range(n_hit-1) for j in range(i+1, n_hit)]
                score = sum(min_dist) / (len(query_tokens) * (len(query_tokens)-1) / 2)
                similarity, doc_id = heap[doc_order]
                heap[doc_order] = (alpha * similarity - (1 - alpha) * score, doc_id)
                # print("similarity: {} , proximity: {}".format(-alpha * similarity, (1 - alpha) * score))
        # sort the scores using a heap
        heapq.heapify(heap)
        answer = []
        score_output = []
        while heap:
            score, doc_id = heapq.heappop(heap)
            if score < 0:
                answer.append(doc_id)
                score_output.append(-score)
            else:
                break
        return answer, score_output

    def __free_text_query(self, tokens, alpha=0.8):
        """
        Returns the query result for a free text query
        :param tokens: the tokens in the query
        :return: a list of doc_ids retrieved by the search engine
        """
        # calculate tfidfs for query tokens
        sorted_tokens, tfs = self.__get_query_tfs(tokens)
        idfs = [self.__get_idf(token) for token in sorted_tokens]
        tfidfs = [(1 + math.log(tfs[i], 10)) * idfs[i] for i in range(len(tfs))]
        return self.__search_similarity(sorted_tokens, tfidfs, alpha=alpha)

    def query(self, query_string, expand=False):
        """
        Returns the query result for a query string
        :param query_string: contains the query
        :param expand: whether use query expansion
        :return: a list of doc_ids retrieved by the search engine
        """
        # tokenize and preprocess the query string
        query_container = self.parser.parse_query(query_string)
        if query_container.q_type == "FreeText":
            if expand:
                expanded_query = self.__expand_query(query_container.data)
                print(expanded_query)
                result, score = self.__free_text_query(expanded_query)
            else:
                result, score = self.__free_text_query(query_container.data)
            # if len(result) > 100:
            #     print(len(result))
            #     rev_score = list(reversed(score))
            #     result = result[:len(result) - bisect.bisect(rev_score, score[0]*0.05)]
            # print(len(result))
            return result
        elif query_container.q_type == "Boolean":
            # Boolean query
            # relevant = set(self.dictionary.doc_ids)

            relevant_dict = {}
            for query_element in query_container.data:
                if expand:
                    expanded_query = self.__expand_query(query_element)
                    result, score = self.__free_text_query(expanded_query, alpha=0.2)
                else:
                    result, score = self.__free_text_query(query_element, alpha=0.2)
                # relevant.intersection_update(result)
                for doc_id_idx, doc_id in enumerate(result):
                    old_score = relevant_dict.get(doc_id, 0)
                    relevant_dict[doc_id] = old_score + score[doc_id_idx]
            result = []
            for doc_id in relevant_dict.keys():
                result.append((relevant_dict[doc_id], doc_id))
            result = sorted(result, reverse=True)
            return [doc_id for (_, doc_id) in result]
        else:
            # ERROR!
            print("ERROR!")
            return []

    def close(self):
        """
        Close the postings file
        :return: nothing
        """
        self.postings_file.close()


if __name__ == "__main__":
    dictionary_file = postings_file = file_of_queries = file_of_output = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
    except getopt.GetoptError as err:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-d':
            dictionary_file = a
        elif o == '-p':
            postings_file = a
        elif o == '-q':
            file_of_queries = a
        elif o == '-o':
            file_of_output = a
        else:
            assert False, "unhandled option"

    if not dictionary_file or not postings_file or not file_of_queries or not file_of_output :
        usage()
        sys.exit(2)

    # start query engine
    query_list = []
    with open(file_of_queries, "r") as f:
        for line in f:
            query_list.append(line)
    with SearchEngine(dictionary_file, postings_file) as engine:
        with open(file_of_output, "w") as f:
            for query_str in query_list:
                query_result = engine.query(query_str, expand=True)
                if query_result:
                    for idx, res in enumerate(query_result[:-1]):
                        f.write(str(res))
                        f.write(" ")
                    f.write(str(query_result[-1]))
                f.write("\n")
