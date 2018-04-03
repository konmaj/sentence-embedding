import numpy as np

from sent_emb.algorithms.fasttext_utility import FastText, FastTextWithoutUnknown
from sent_emb.algorithms.glove_embeddings_mean import WordVectorsMean

from sent_emb.algorithms.simpleSVD import SimpleSVD, ExternalProbFocusUnknown


class FastTextMean(WordVectorsMean):
    def __init__(self):
        self.word_embeddings = FastText()


class FastTextMeanWithoutUnknown(WordVectorsMean):
    def __init__(self):
        self.word_embeddings = FastTextWithoutUnknown()


class FastTextSVD(SimpleSVD):
    def __init__(self, param_a=0.001, prob=ExternalProbFocusUnknown()):
        super(FastTextSVD, self).__init__(FastText(), param_a, prob)
