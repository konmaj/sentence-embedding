from abc import ABC, abstractmethod
from collections import namedtuple


class DataSet(ABC):
    """
    DataSet class, which represents all data Algorithm should know beforehand about it.
    """

    @abstractmethod
    def word_set(self):
        """
        Should be lazy if computation may be long
        :return: Set of all words used in sentences in this DataSet
        """
        pass

    @abstractmethod
    def tokenizer_name(self):
        """
        :return: Name of the tokenizer (may be used eg. for filenames)
        """
        pass


class WordEmbedding(ABC):
    """
    Abstract class, which represents embeddings for words
    """

    @abstractmethod
    def get_dim(self):
        """
        :return: size of the vector of the word embedding
        """
        pass

    @abstractmethod
    def get_resources(self, dataset):
        """
        Must be called once before getting embeddings,
        downloads or prepares needed resources
        :param dataset: DataSet class, for  which we prepare resources
        """
        pass

    @abstractmethod
    def embeddings(self, sents):
        """
        Main function, which computes vectors for all words used in task
        :param sents: list of sentences
        :return: dictionary of embeddings for words used in sentences
        """
        pass


SentPair = namedtuple('SentPair', ['sent1', 'sent2'])

SentPairWithGs = namedtuple('SentPairWithGs', SentPair._fields + ('gs',))


class BaseAlgorithm(ABC):
    """
    Base abstract class, which represents algorithms, which compute sentence embeddings.

    Note - following call:
        SubclassName()
    should yield valid (but maybe not optimally tuned) object of SubclassName
    (for the purpose of smoketest).
    """

    @abstractmethod
    def get_resources(self, task):
        """
        Called once before all evaluations. Prepares external resources
        (downloads data, crops files for better performace).
        :param task: Task object
        """
        pass


    @abstractmethod
    def fit(self, sents):
        """
        Prepares 'self' object on training data.

        Should be called before first call to transform() method.

        Every call to fit() method has to:
        1) clear the state of algorithm,
        2) train model from scratch using only 'sents' passed as an argument
           (data from previous calls of fit() has to be ignored).

        sents: sequence of sentences as in sent_emb.evaluation.sts.read_sts_input() function's result
        returns: None
        """
        pass

    @abstractmethod
    def transform(self, sents):
        """
        Computes embeddings of given sentences.

        sents: sequence of sentences as in sent_emb.evaluation.sts.read_sts_input() function's result
        returns: numpy 2-D array of sentence embeddings
        """
        pass


# Auxiliary functions for operating on SentPair and SentPairWithGs lists

def zip_sent_pairs_with_gs(sent_pairs, gold_standards):
    assert len(sent_pairs) == len(gold_standards)
    return [SentPairWithGs(*pair, gs=gs) for pair, gs in zip(sent_pairs, gold_standards)]


def flatten_sent_pairs(sent_pairs):
    return [sent for sent_pair in sent_pairs for sent in [sent_pair.sent1, sent_pair.sent2]]
