from abc import abstractmethod
import nltk


class Preprocessing:
    @abstractmethod
    def tokenize(self, sent):
        pass

    def name(self):
        pass


class PreprocessingNltk(Preprocessing):
    def tokenize(self, sent):
        return [ch.lower() for ch in nltk.tokenize.word_tokenize(sent)]

    def name(self):
        return 'simpleNLTK'


class PreprocessingStanford(Preprocessing):
    def __init__(self):
        self.tokenizer = nltk.tokenize.stanford.StanfordTokenizer()

    def tokenize(self, sent):
        return [ch.lower() for ch in self.tokenizer.tokenize(sent)]

    def name(self):
        return 'StanfordCoreNLPTokenizer'
