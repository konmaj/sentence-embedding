from sent_emb.algorithms.fasttext_utility import FastText, FastTextWithoutUnknown, FastTextWithGloVeLength
from sent_emb.algorithms.glove_embeddings_mean import WordVectorsMean

from sent_emb.algorithms.simpleSVD import SimpleSVD, ExternalProbFocusUnknown, ExternalProb


class FastTextMean(WordVectorsMean):
    def __init__(self, length='normal'):
        if length == 'normal':
            self.word_embeddings = FastText()
        elif length == 'glove':
            self.word_embeddings = FastTextWithGloVeLength()
        else:
            print('Invalid length={} argument in FastTextMean constructor'.format(length))
            assert False


class FastTextMeanWithoutUnknown(WordVectorsMean):
    def __init__(self):
        self.word_embeddings = FastTextWithoutUnknown()


class FastTextSVD(SimpleSVD):
    def __init__(self, param_a=0.001, prob=ExternalProbFocusUnknown(), length='normal'):
        if length == 'normal':
            fasttext = FastText()
        elif length == 'glove':
            fasttext = FastTextWithGloVeLength()
        else:
            print('Invalid length={} argument in FastTextSVD constructor'.format(length))
            assert False
        super(FastTextSVD, self).__init__(fasttext, param_a, prob)


class FastTextSVDWithoutUnknown(SimpleSVD):
    def __init__(self, param_a=0.001, prob=ExternalProb()):
        fasttext = FastTextWithoutUnknown()
        super(FastTextSVDWithoutUnknown, self).__init__(fasttext, param_a, prob)
