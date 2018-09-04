from abc import abstractmethod

import pickle
import keras
import nltk
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

from sent_emb.algorithms.path_utility import OTHER_RESOURCES_DIR
from sent_emb.algorithms.seq2seq.preprocessing import preprocess_sents

from sent_emb.evaluation.model import BaseAlgorithm, zip_sent_pairs_with_gs, SentPair
from sent_emb.evaluation.sts_eval import eval_sts_all
from sent_emb.evaluation.sts_read import STS, read_train_set, tokens

from nltk.corpus import brown, webtext, reuters, BNCCorpusReader

WEIGHTS_PATH = OTHER_RESOURCES_DIR.joinpath('seq2seq', 'weights')


def get_weights_paths(name):
    return (WEIGHTS_PATH.joinpath('{}.h5'.format(name)),
            WEIGHTS_PATH.joinpath('{}_enc.h5'.format(name)))


def load_model_weights(name, complete_model, encoder_model, force_load=True):
    all_weights_path, encoder_weights_path = get_weights_paths(name)

    # Try to load saved model from disk.
    if all_weights_path.exists() and encoder_weights_path.exists():
        print('Loading weights from files:\n{}\n{}'.format(all_weights_path, encoder_weights_path))
        complete_model.load_weights(str(all_weights_path))
        encoder_model.load_weights(str(encoder_weights_path))
    else:
        if force_load:
            error_msg = \
                '''ERROR: Weights not found and force_load==True
                       If you really want to create new files with weights, please add
                       --alg-kwargs='{"force_load": false}' as a param to the script.
                '''
            print(error_msg)
            assert False
        else:
            print('Weights not found - algorithm will start with not trained model.')


def save_model_weights(name, complete_model, encoder_model):
    WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)

    all_weights_path, encoder_weights_path = get_weights_paths(name)

    complete_model.save_weights(str(all_weights_path))
    encoder_model.save_weights(str(encoder_weights_path))

    print('Weights are saved in files:\n{}\n{}'.format(all_weights_path, encoder_weights_path))


def use_gpu_if_present():
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)


class Seq2Seq(BaseAlgorithm):
    """Abstract class intended to be an interface for Seq2Seq algorithms."""

    def __init__(self, word_embedding, sent_emb_dim):
        """
        :param word_embedding: source of word embeddings (object of WordEmbedding class)

        :param sent_emb_dim: dimension of sentence embeddings, which this algorithm will produce
        """
        self.word_embedding = word_embedding
        self.sent_emb_dim = sent_emb_dim

        self.complete_model = None
        self.encoder_model = None

        use_gpu_if_present()

    def _check_members_presence(self):
        """
        Checks whether all needed members are present in `self` object.

        It is recommended to call this method at the end of __init__ method of subclasses.
        """
        members = [self.word_embedding, self.complete_model, self.encoder_model, self.sent_emb_dim]
        for member in members:
            if member is None:
                error_msg = '''
                Seq2Seq: One of important class members is null.
                See docstring of Seq2Seq class for information, what members are needed.'''
                print(error_msg)
                assert False

    def fit(self, sents):
        """
        We don't want to train model during evaluation - it would take too much time.
        """
        pass

    @abstractmethod
    def improve_weights(self, sent_pairs, epochs, **kwagrs):
        """
        Real training of the model.

        Implementations should always save weights of model after training.

        sent_pairs: list of SentPairWithGs objects
                    (note - this type is different from the type of `sents` in `fit` method)

        epochs: number of epochs to train

        returns: None
        """
        pass

    def transform(self, sents):
        self._check_members_presence()

        sents_vec = preprocess_sents(sents, self.word_embedding)

        print("Shape of sentences after preprocessing:", sents_vec.shape)
        assert sents_vec.shape[0] == len(sents) and sents_vec.shape[2] == self.word_embedding.get_dim()

        embs = self.encoder_model.predict(sents_vec)

        assert embs.shape == (sents_vec.shape[0], self.sent_emb_dim)

        return embs

    def get_resources(self, dataset):
        self.word_embedding.get_resources(dataset)


def process_corpus(corp, tokenizer):
    sent_pairs = []

    raw_sents = []
    for id in corp.fileids():
        raw_sents += [' '.join(sent) for sent in corp.sents(id)]
    print("Sentences fetched")

    sents = tokens(tokenizer, raw_sents)

    base = 0

    for id in corp.fileids():
        for i in range(1, len(corp.sents(id)) - 1):
            # sent_pairs.append(SentPair(sents[base + i], (sents[base + i - 1], sents[base + i + 1])))
            sent_pairs.append(SentPair(sents[base + i], sents[base + i + 1]))
        base += len(corp.sents(id))

    print("Sentences tokenized")

    return sent_pairs


def read_corpus(tokenizer):
    nltk.download('brown')
    nltk.download('webtext')
    nltk.download('reuters')
    # bnc = BNCCorpusReader(root="../../sentence-embedding/resources/other/2554/download/Texts",
    #                              fileids=r'[A-K]/\w*/\w*\.xml')

    sent_pairs = []

    for corp in [brown]:
        sent_pairs += process_corpus(corp, tokenizer)

    print("Corpus ready")

    # raw_sents += [' '.join(sent) for sent in webtext.sents()]
    # print(raw_sents[0])

    # sents = tokens(tokenizer, raw_sents)
    # print(sents[0])

    # sent_pairs = [SentPair(sent, sent) for sent in sents]

    gs_sents = zip_sent_pairs_with_gs(sent_pairs, [0] * len(sent_pairs))
    print(gs_sents[0])

    return gs_sents


def get_words(sents, frac):
    sents = [gs.sent1 for gs in sents] + [gs.sent2 for gs in sents]
    # print(sents[:5])
    print("Flat text size:", len(sents)*len(sents[0]))

    vectorizer = CountVectorizer(binary=True, token_pattern=r"\b\w+\b")
    vectorizer.fit([' '.join(sent) for sent in sents])

    voc = len(vectorizer.vocabulary_)
    print("Vocabulary size:", voc)

    # vectorizer = CountVectorizer(binary=True, token_pattern=r"\b\w+\b", max_features=int(voc*frac), max_df=0.05)
    vectorizer = CountVectorizer(binary=True, token_pattern=r"\b\w+\b", max_features=int(voc*frac))
    vectorizer.fit([' '.join(sent) for sent in sents])

    print("Final vocabulary:", len(vectorizer.vocabulary_))

    return vectorizer


def improve_model(algorithm, tokenizer, epochs=1, eval_interval=None, add_corpus=False):
    """
    Runs training of Seq2Seq `algorithm` on STS16 training set.

    epochs: number of epochs to train the model

    eval_interval: specifies how often evaluation should be run
        if None, then no evaluation will be run

        otherwise evaluation will be run after:
            eval_interval[0], eval_interval[0] + eval_interval[1], ...
        epochs of training, until the `epochs` limit will be reached

        when there is not enough elements in `eval_interval` list, then `eval_interval[-1]` is used.
    """
    assert isinstance(algorithm, Seq2Seq)
    algorithm.get_resources(STS(tokenizer))

    print('Reading training set...')
    sent_pairs = []
    # sent_pairs = read_train_set(16, tokenizer)
    # print(sent_pairs[0])
    if add_corpus:
        print('Reading additional corpus...')
        fname = 'corp_1.dat'
        try:
            with open(fname, 'rb') as f:
                corpus = pickle.load(f)
        except FileNotFoundError:
            corpus = read_corpus(tokenizer)
            with open(fname, 'wb') as f:
                pickle.dump(corpus, f)
        sent_pairs = corpus
    print('...done.')

    if eval_interval is None:
        algorithm.improve_weights(sent_pairs, epochs)
    else:
        years_to_eval = [12, 15]
        # years_to_eval = [12, 13, 14, 15, 16]

        completed_epochs = 0
        for interval in eval_interval:
            epochs_to_run = min(epochs - completed_epochs, interval)
            if epochs_to_run == 0:
                break

            algorithm.improve_weights(sent_pairs, epochs_to_run)
            completed_epochs += epochs_to_run

            eval_sts_all(algorithm, tokenizer, years_choose=years_to_eval, training=False)

        while completed_epochs < epochs:
            epochs_to_run = min(epochs - completed_epochs, eval_interval[-1])

            algorithm.improve_weights(sent_pairs, epochs_to_run)
            completed_epochs += epochs_to_run

            eval_sts_all(algorithm, tokenizer, years_choose=years_to_eval, training=False)
