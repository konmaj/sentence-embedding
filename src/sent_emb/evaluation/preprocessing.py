from abc import ABC, abstractmethod

import spacy
import nltk
import re


class Preprocessing(ABC):
    @abstractmethod
    def tokenize(self, sent):
        pass

    def name(self):
        pass


class PreprocessingNltk(Preprocessing):
    def __init__(self):
        self.tokenizer = nltk.tokenize.stanford.StanfordTokenizer()

    def tokenize(self, sent):
        return [ch.lower() for ch in nltk.tokenize.word_tokenize(sent)]

    def name(self):
        return 'simpleNLTK'


class PreprocessingStanford(Preprocessing):
    def __init__(self):
        self.tokenizer = nltk.tokenize.stanford.StanfordTokenizer()
        self._should_merge = True

    def tokenize(self, sent):
        return self.tokenizer.tokenize(sent)

    def name(self):
        return 'StanfordCoreNLPTokenizer'


class PreprocessingStanfordLowercase(Preprocessing):
    def __init__(self):
        self.tokenizer = nltk.tokenize.stanford.StanfordTokenizer()
        self._should_merge = True

    def tokenize(self, sent):
        return [ch.lower() for ch in self.tokenizer.tokenize(sent)]

    def name(self):
        return 'StanfordCoreNLPTokenizerLowercase'


class PreprocessingStanfordExtra(Preprocessing):
    def __init__(self):
        jar = '/home/michal/Projekty/sentence-embedding/resources/stanford/stanford-postagger-full-2015-04-20/stanford-postagger.jar'
        self.tokenizer = nltk.tokenize.stanford.StanfordTokenizer(jar)
        self._should_merge = True

    def tokenize(self, sent):
        sent = re.sub('#', ' ', sent)
        sent = re.sub('/', ' ', sent)
        sent = re.sub('_', ' ', sent)
        return [ch.lower() for ch in self.tokenizer.tokenize(sent)]

    def name(self):
        return 'StanfordCoreNLPTokenizerExtra'


POS_TAGS = ['ADJ', 'ADV', 'PART', 'NOUN', 'PROPN', 'VERB']

class PreprocessingSpacy(Preprocessing):
    def __init__(self):
        self.nlp = spacy.load('en')

    def tokenize(self, sent):
        sent = re.sub('#', ' ', sent)
        sent = re.sub('_', ' ', sent)
        doc = self.nlp(sent)
        res = [w.text.lower() for w in doc if w.pos_ in POS_TAGS]

        if res == []:
            return [sent[0].lower()]
        return res

    def name(self):
        return 'SpacyTokenizer'
