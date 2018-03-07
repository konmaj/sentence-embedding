from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    '''
    Base abstract class, which represents algorithms, which compute sentence embeddings.
    '''

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
