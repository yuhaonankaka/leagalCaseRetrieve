import pickle


class Dictionary(object):

    def __init__(self, vocabulary, doc_ids):
        """
        :param vocabulary: a python list containing the vocabulary used by the dictionary
                            (dictionary will NOT sort the vocabulary list)
        :param doc_ids: a list of all docIDs
        """
        # index to string mapping (termID -> token)
        self.itos = vocabulary
        # string to index mapping (token -> termID)
        self.stoi = {}
        self.__construct_stoi()
        # index to pointers in file
        self.itop = None
        self.doc_ids = doc_ids
        self.doc_order = {}
        for idx, doc_id in enumerate(doc_ids):
            self.doc_order[doc_id] = idx
        self.doc_len = None
        self.dfs = None
        self.cfs = None

    def __getitem__(self, key):
        """
        Implements [] operator, looks up file pointer of a token
        :param key: the token to look up
        :return: file pointer
        """
        if key in self.stoi:
            return self.itop[self.stoi[key]]
        else:
            return None

    def __construct_stoi(self):
        """
        Internal method for constructing stoi mapping from itos mapping
        :return: Nothing
        """
        for idx, token in enumerate(self.itos):
            self.stoi[token] = idx

    def get_token(self, idx):
        """
        Looks up the token of a termID in dictionary
        :param idx: the index to loop up
        :return: token
        """
        return self.itos[idx]

    def add_dfs(self, dfs):
        self.dfs = dfs

    def add_doc_len(self, doc_len):
        self.doc_len = doc_len

    def add_pointers(self, ptrs):
        self.itop = ptrs

    def add_cfs(self, cfs):
        self.cfs = cfs

    def save(self, path):
        """
        Save the dictionary itself to disk
        :param path: the path to the save file
        :return: nothing
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
