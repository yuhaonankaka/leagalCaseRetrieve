#!/usr/bin/python
import math
import nltk
import sys
import getopt
import heapq
import pickle
import pandas as pd
from dictionary import Dictionary
from query_parser import Parser


def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")


class Indexer(object):

    def __init__(self, preprocess=None, tokenize=None):
        self.dictionary = None
        self.postings = None
        self.positions = None
        self.byte_repr = b""
        self.repr_ptrs = []
        self.vocabulary = None
        self.dfs = None
        if preprocess:
            self.__preprocess = preprocess
        if tokenize:
            self.__tokenize = tokenize

    def __tokenize(self, document):
        return nltk.word_tokenize(document)

    def __preprocess(self, tokens):
        return [token.lower() for token in tokens]

    def __build_postings(self, doc_tokens):
        """
        Builds postings from document tokens.
        :param doc_tokens: list of document tokens, where each element is a list of tokens in a document
        :return: post_dict: maps term -> (list of postings, list of term frequency)
                 df_dict: maps term -> document frequency
                 doc_len_dict: maps docID -> document vector norm
        """
        post_dict = {}
        tf_dict = {}
        pos_dict = {}
        df_dict = {}
        cf_dict = {}
        doc_len_dict = {}
        count = 0
        # get tf information
        for doc in doc_tokens:
            doc_id, term_list = doc
            tf_dict[doc_id] = {}
            pos_dict[doc_id] = {}
            doc_len_dict[doc_id] = 0
            # tfs
            for pos, term in enumerate(term_list):
                if term in cf_dict:
                    cf_dict[term] += 1
                else:
                    cf_dict[term] = 0
                tf_dict[doc_id][term] = tf_dict[doc_id].get(term, 0) + 1
                orig_pos = pos_dict[doc_id].get(term, [])
                orig_pos.append(pos)
                pos_dict[doc_id][term] = orig_pos
                # postings
                # if postings list is empty or does not contain this doc_id
                if term not in post_dict:
                    post_dict[term] = [doc_id]
                elif post_dict[term][-1] != doc_id:
                    post_dict[term].append(doc_id)
            count += 1
            print("Building postings for doc {} out of {}...".format(count, len(doc_tokens)))

        # build vocabulary
        self.vocabulary = [term for term in post_dict.keys()]

        # restructure tf for each term in each documents
        for term in self.vocabulary:
            # calculate df
            df_dict[term] = len(post_dict[term])
            # get tf
            tfs = []
            for doc_id in post_dict[term]:
                tfidf = 1 + math.log(tf_dict[doc_id][term], 10)
                tfs.append(tf_dict[doc_id][term])
                doc_len_dict[doc_id] += tfidf*tfidf
            post_dict[term] = (post_dict[term], tfs)

        for key, value in doc_len_dict.items():
            doc_len_dict[key] = math.sqrt(value)
        return post_dict, df_dict, doc_len_dict, pos_dict, cf_dict

    def __save_byte_repr(self, postings_path, encoding_length=3):
        """
        Construct the byte representation of this index. Entries are stored consecutively, and each entry is formatted
        as follows:
            [length of entry (number of items), term_id, length of postings (integer count),
            postings: docID1, docID2, ..., tf1, tf2 ...,
            positions: length1, pos1_1, pos1_2, ..., length2, pos2_1, ...]
        :param encoding_length: integer encoding length
        :return: nothing
        """

        def to_byte_rep(x): return x.to_bytes(encoding_length, byteorder="little")

        file_ptr = 0
        ptrs = {}
        with open(postings_path, "wb") as f:
            for term_id, token in enumerate(self.vocabulary):
                ptrs[term_id] = file_ptr
                doc_ids, tfs = self.postings[token]
                # construct byte representation of each entry
                # calculate length of positions
                len_positions = 0
                for doc_id in doc_ids:
                    len_positions += len(self.positions[doc_id][token]) + 1
                # length of entry
                f.write(to_byte_rep(len(doc_ids)*2 + 3 + len_positions))
                # term_id
                f.write(to_byte_rep(term_id))
                # length of postings
                f.write(to_byte_rep(len(doc_ids)))
                # postings
                for doc_id in doc_ids:
                    f.write(to_byte_rep(doc_id))
                # tfs
                for tf in tfs:
                    f.write(to_byte_rep(tf))
                # positions
                for doc_id in doc_ids:
                    pos_list = self.positions[doc_id][token]
                    f.write(to_byte_rep(len(pos_list)))
                    for pos in pos_list:
                        f.write(to_byte_rep(pos))
                file_ptr += (len(doc_ids) * 2 + 3 + len_positions) * encoding_length
        self.repr_ptrs = ptrs

    def index(self, input_file):
        """
        Construct the reverse index of all files in input_file_dir, applicable to Reuters Corpus, where file names are
        the corresponding docIDs
        :param input_file: the path to the directory containing the files to index
        :return: nothing
        """
        # get training file names
        doc_ids = []
        # construct index
        document_tokens = []
        csv_data = pd.read_csv(input_file)
        # for i in range(csv_data.shape[0]):
        for i in range(10):
            document = csv_data.iloc[i]
            doc_id = int(document.document_id)
            doc_ids.append(doc_id)
            # read file contents
            tokens = []
            tokens += self.__preprocess(self.__tokenize(document.content))
            document_tokens.append((doc_id, tokens))
            print("Processing {} out of {} lines...".format(i + 1, csv_data.shape[0]))

        # build postings from (docID, token) pairs
        postings_dict, df_dict, doc_len_dict, pos_dict, cf_dict = self.__build_postings(document_tokens)
        self.vocabulary = sorted(postings_dict.keys())
        self.postings = postings_dict
        self.positions = pos_dict
        self.dfs = df_dict

        self.dictionary = Dictionary(self.vocabulary, doc_ids)
        self.dictionary.add_dfs(self.dfs)
        self.dictionary.add_doc_len(doc_len_dict)
        self.dictionary.add_cfs(cf_dict)

    def save(self, postings_path, dictionary_path):
        """
        Saves the byte representation of postings and dictionary to file.
        :param postings_path: the path of postings file
        :param dictionary_path: the path of dictionary file
        :return: nothing
        """
        # construct byte representation for postings
        self.__save_byte_repr(postings_path)
        self.dictionary.add_pointers(self.repr_ptrs)
        self.dictionary.save(dictionary_path)


if __name__ == "__main__":
    input_directory = output_file_dictionary = output_file_postings = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
    except getopt.GetoptError as err:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-i':  # input directory
            input_directory = a
        elif o == '-d':  # dictionary file
            output_file_dictionary = a
        elif o == '-p':  # postings file
            output_file_postings = a
        else:
            assert False, "unhandled option"

    if not input_directory or not output_file_postings or not output_file_dictionary:
        usage()
        sys.exit(2)

    # construct index
    parser = Parser()
    indexer = Indexer(preprocess=parser.preprocess, tokenize=parser.tokenize)
    indexer.index(input_directory)

    # save postings to file
    indexer.save(output_file_postings, output_file_dictionary)
