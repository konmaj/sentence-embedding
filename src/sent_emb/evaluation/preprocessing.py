from abc import abstractmethod
import nltk


class Preprocessing():
    @abstractmethod
    def tokenize(self, sent):
        pass


class PreprocessingNltk():
    @staticmethod
    def tokenize(sent):
        return [ch.lower() for ch in nltk.tokenize.word_tokenize(sent)]


class PreprocessingStanford():
    def __init__(self):
        self.tokenizer = nltk.tokenize.stanford.StanfordTokenizer()

    def tokenize(self, sent):
        return [ch.lower() for ch in self.tokenizer.tokenize(sent)]
