import numpy as np

from sent_emb.algorithms.fasttext_utility import read_file, FASTTEXT_UNKNOWN, FASTTEXT_CROPPED
from sent_emb.algorithms.unknown import UnknownVector
from sent_emb.evaluation.model import BaseAlgorithm
from pathlib import Path

from sent_emb.algorithms.simpleSVD import SimpleSVD, ExternalProbFocusUnknown

def normalize(vec):
    return vec / np.linalg.norm(vec)


def get_unknown_file():
    return Path('/', 'opt', 'resources', 'embeddings', 'fasttext', 'queries.txt')


class FastText():
    def embeddings(self, used):
        answer = {}

        def process(word, vec, _):
            if word in used:
                answer[word] = normalize(vec)

        read_file(FASTTEXT_CROPPED, process)
        read_file(FASTTEXT_UNKNOWN, process)
        return answer


class FastTextWithoutUnknown():
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


class FastTextMean(BaseAlgorithm):
    def __init__(self, fast_text=FastText()):
        self.fastText = fast_text

    def fit(self, _):
        return self

    def transform(self, sents):
        used = set()
        for sent in sents:
            for word in sent:
                used.add(word)

        wordvec = self.fastText.embeddings(used)

        for x in wordvec:
            print(wordvec[x].shape[0])
            result = np.zeros((sents.shape[0], wordvec[x].shape[0]), dtype=np.float)
            break

        count = np.zeros((sents.shape[0], 1))

        for idx, sent in enumerate(sents):
            for word in sent:
                result[idx] += normalize(wordvec[word])
                count[idx][0] += 1

        result /= count

        return result


class FastTextMeanWithoutUnknown(BaseAlgorithm):
    def __init__(self):
        self.model = FastTextMean(FastTextWithoutUnknown())

    def fit(self, sents):
        return self.model.fit(sents)

    def transform(self, sents):
        return self.model.transform(sents)


class FastTextSVD(BaseAlgorithm):
    def __init__(self, param_a=0.001, prob=ExternalProbFocusUnknown()):
        self.simpleSVD = SimpleSVD(FastText(), param_a, prob)

    def fit(self, sents):
        return self.simpleSVD.fit(sents)

    def transform(self, sents):
        return self.simpleSVD.transform(sents)
