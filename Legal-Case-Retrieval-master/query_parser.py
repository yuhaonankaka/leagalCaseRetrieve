import nltk
nltk.data.path.append('./nltk_data')
from nltk.corpus import wordnet as wn

from nltk.stem.porter import *
from nltk.corpus import stopwords
from query import Query


class Parser(object):
    def __init__(self):
        self.operators = ["NOT", "AND", "OR"]
        self.stemmer = PorterStemmer()
        self.stopWords = set(stopwords.words('english'))

    def tokenize(self, document):
        return nltk.word_tokenize(document)

    def preprocess(self, tokens):
        result = []
        for token in tokens:
            if token != "AND":
                # if not token.isnumeric() and token not in self.stopWords:
                #     # if token not in self.stopWords:
                result.append(self.stemmer.stem(token.lower()))
            else:
                result.append(token)
        return result

    def parse_query(self, query):
        processed_tokens = self.preprocess(self.tokenize(query))
        if "``" not in processed_tokens and "AND" not in processed_tokens:  # this is free text
            return Query("FreeText", processed_tokens)
        # this is Boolean Text
        check_result = self.check_query(self.tokenize(query))
        if not check_result:
            return Query("Error", processed_tokens)
        result = []
        temp_phrase = []
        start = False  # start of the phrase
        for token in processed_tokens:
            if token == "``":
                start = True
            elif token == "''":
                start = False
                result.append(temp_phrase)
                temp_phrase = []
            elif token != "AND" and start:
                temp_phrase.append(token)
            elif token != "AND" and not start:
                result.append([token])
        return Query("Boolean", result)

    def check_query(self, tokens):
        result = []
        temp_phrase = []
        start = False  # start of the phrase
        for token in tokens:
            if token == "``":
                start = True
            elif token == "''":
                start = False
                result.append(temp_phrase)
                temp_phrase = []
            elif token != "AND" and start:
                temp_phrase.append(token)
            elif token != "AND" and not start:
                result.append([token])
            else:
                result.append([token])
        count = 1
        if not result:
            return False
        for token in result:
            if count % 2 == 0:
                if not token or token[0] != 'AND':
                    return False
            count += 1
        return True

    '''
    params: 1.oriWord:String 2. tarWord: String
    return: max similarity: float
    '''
    def bestSimilarity(self, ori_word, tar_word):
        ori_word = ori_word.lower()
        tar_word = tar_word.lower()
        syns_ori = wn.synsets(ori_word)
        ori_words = []
        tar_words = []
        for syn in syns_ori:
            if syn.lemmas()[0].name() == ori_word:
                # print(syn)
                ori_words.append(syn)
        syns_tar = wn.synsets(tar_word)
        for syn in syns_tar:
            if syn.lemmas()[0].name() == tar_word:
                tar_words.append(syn)
        max_sim = 0
        for ori_word in ori_words:
            for tar_word in tar_words:
                if tar_word.wup_similarity(ori_word) is not None and tar_word.wup_similarity(ori_word) > max_sim:
                    max_sim = tar_word.wup_similarity(ori_word)
        return max_sim

    '''
    params: 1.oriWord:String 2. tarWordList: String[] 3. k: Int
    return: List of pair (word:String, similarity:float)
    '''
    def top_k_similarity(self, oriWord, tarWordList, k):
        valueMap = {}
        for tarWord in tarWordList:
            sim = self.bestSimilarity(oriWord, tarWord)
            valueMap[tarWord] = sim
        result = sorted(valueMap.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        # print(result[0:k])
        return result[0:k]


if __name__ == "__main__":
    parser = Parser()
    token = "gild"
    syn_name = []
    for synset in wn.synsets(token):
        syn_name.append(parser.preprocess([synset.name().split(".")[0]])[0])
    print(token)
    result = parser.top_k_similarity(token, syn_name, 10)
    print(result)
