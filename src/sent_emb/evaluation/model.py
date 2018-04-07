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
        :return: Name of the tokenizer (may be used e.g. for filenames)
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
        :param dataset: DataSet class, for which we prepare resources
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
"""
Class, which represents pair of tokenized sentences from STS task.
Each sentence is intended to be a list of words (list of strings), but the class does not control
type of sentences.
"""

SentPairWithGs = namedtuple('SentPairWithGs', SentPair._fields + ('gs',))
"""
Class, which represents pair of tokenized sentences from STS task with corresponding gold standard
similarity score.
Gold standard is intended to be `float` in range [0; 5] or None, when missing. The class does not
control these conditions.
"""


class BaseAlgorithm(ABC):
    """
    Base abstract class, which represents algorithms, which compute sentence embeddings.

    Note - following call:
        SubclassName()
    should construct valid (but maybe not optimally tuned) object of SubclassName
    (for the purpose of smoke test).
    """

    @abstractmethod
    def get_resources(self, dataset):
        """
        Should be called once before all evaluations. Prepares external resources
        (downloads data, crops files for better performace).
        :param dataset: DataSet object
        """
        pass


    @abstractmethod
    def fit(self, sents):
        """
        Prepares `self` object on training data.

        Should be called before the first call to transform() method.

        Every call to fit() method has to:
        1) clear the state of algorithm,
        2) train model from scratch using only 'sents' passed as an argument
           (data from previous calls of fit() has to be ignored).

        sents: list of sentences in training set - each sentence is a list of words (list of strings).
        returns: None
        """
        pass

    @abstractmethod
    def transform(self, sents):
        """
        Computes embeddings of given sentences.

        sents: list of sentences to compute embeddings (in the same format as in `fit` method)
        returns: numpy 2-D array of sentence embeddings (second dimension may be arbitrary)
        """
        pass


# Auxiliary functions for operating on lists of SentPair or SentPairWithGs

def flatten_sent_pairs(sent_pairs):
    """
    Converts list of SentPair or SentPairWithGs to list of sentences
    (sentences from single pair are adjacent in resulting list).

    :param sent_pairs: list of SentPair or SentPairWithGs objects
    :return: list of list of strings
    """
    return [sent for sent_pair in sent_pairs for sent in [sent_pair.sent1, sent_pair.sent2]]


def zip_sent_pairs_with_gs(sent_pairs, gold_standards):
    """
    Zips list of SentPairs with list of corresponding gold standard scores.

    :param sent_pairs: list of SentPairs objects
    :param gold_standards: list of gold standard scores
    :return: list of SentPairWithGs objects
    """
    assert len(sent_pairs) == len(gold_standards)

    return [SentPairWithGs(*pair, gs=gs) for pair, gs in zip(sent_pairs, gold_standards)]


def get_gold_standards(sent_pairs_with_gs):
    return [sent_pair.gs for sent_pair in sent_pairs_with_gs]
