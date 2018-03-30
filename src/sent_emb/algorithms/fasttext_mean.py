import numpy as np

from sent_emb.algorithms.fasttext_utility import read_file, FASTTEXT_UNKNOWN, FASTTEXT_CROPPED, get_fasttext_resources
from sent_emb.algorithms.unknown import UnknownVector, NoUnknown
from sent_emb.algorithms.glove_embeddings_mean import WordVectorsMean
from sent_emb.evaluation.model import BaseAlgorithm
from pathlib import Path

from sent_emb.algorithms.simpleSVD import SimpleSVD, ExternalProbFocusUnknown


def normalize(vec):
    return vec / np.linalg.norm(vec)


def get_unknown_file():
    return Path('/', 'opt', 'resources', 'embeddings', 'fasttext', 'queries.txt')


class FastText():
    def get_dim(self):
        return 300

    def get_resources(self, task):
        get_fasttext_resources(task)

    def embeddings(self, used):
        answer = {}

        def process(word, vec, _):
            if word in used:
                answer[word] = normalize(vec)

        read_file(FASTTEXT_CROPPED, process)
        read_file(FASTTEXT_UNKNOWN, process)
        return answer


class FastTextWithoutUnknown():
    def get_dim(self):
        return 300

    def get_resources(self, task):
        get_fasttext_resources(task)

    def embeddings(self, used):
        answer = {}
        unknown = UnknownVector(300)

        def process(word, vec, _):
            unknown.see(word, vec)
            if word in used:
                answer[word] = vec

        read_file(FASTTEXT_CROPPED, process)

        for word in used:
            if word not in answer:
                answer[word] = unknown.get(word)

        return answer


class FastTextMean(WordVectorsMean):
    def __init__(self):
        self.word_embeddings = FastText()


class FastTextMeanWithoutUnknown(BaseAlgorithm):
    def __init__(self):
        self.word_embeddings = FastTextWithoutUnknown()


class FastTextSVD(BaseAlgorithm):
    def __init__(self, param_a=0.001, prob=ExternalProbFocusUnknown()):
        self.simpleSVD = SimpleSVD(FastText(), param_a, prob)

    def fit(self, sents):
        return self.simpleSVD.fit(sents)

    def transform(self, sents):
        return self.simpleSVD.transform(sents)
