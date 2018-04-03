from abc import ABC, abstractmethod


class Task(ABC):
    '''
    Task class, which represents all data Algorithm should know beforehand about it.
    '''

    @abstractmethod
    def word_set(self):
        pass

    def tokenizer_name(self):
        pass


class WordEmbedding(ABC):
    '''
    Abstract class, which represents embeddings for words
    '''

    def get_dim(self):
        '''
        :return: size of embeddings
        '''
        pass


    def get_resources(self, task):
        '''
        Called once downloads or prepares needed resources
        :param task: Task class
        '''
        pass

    def embeddings(self, sents):
        '''
        Main function, which computes vectors for all words used in task
        :param sents: list of sentences
        :return: dictionary of embeddings for words used in sentences
        '''
        pass


class BaseAlgorithm(ABC):
    '''
    Base abstract class, which represents algorithms, which compute sentence embeddings.

    Note - following call:
        SubclassName()
    should yield valid (but maybe not optimally tuned) object of SubclassName
    (for the purpose of smoketest).
    '''

    @abstractmethod
    def get_resources(self, task):
        '''
        Called once before all evaluations. Prepares external resources
        (downloads data, crops files for better performace).
        :param task: Task object
        '''
        pass


    @abstractmethod
    def fit(self, sents):
        '''
        Prepares 'self' object on training data.

        Should be called before first call to transform() method.

        Every call to fit() method has to:
        1) clear the state of algorithm,
        2) train model from scratch using only 'sents' passed as an argument
           (data from previous calls of fit() has to be ignored).

        sents: sequence of sentences as in sent_emb.evaluation.sts.read_sts_input() function's result
        returns: None
        '''
        pass

    @abstractmethod
    def transform(self, sents):
        '''
        Computes embeddings of given sentences.

        sents: sequence of sentences as in sent_emb.evaluation.sts.read_sts_input() function's result
        returns: numpy 2-D array of sentence embeddings
        '''
        pass
