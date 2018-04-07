import numpy as np
import math

import keras
from keras.models import Model
from keras.layers import Input, Dense, GRU

import tensorflow as tf

from sent_emb.algorithms.glove_utility import GloVeSmall
from sent_emb.algorithms.path_utility import RESOURCES_DIR

from sent_emb.evaluation.model import BaseAlgorithm, flatten_sent_pairs
from sent_emb.evaluation.sts_read import STS, read_train_set


BATCH_SIZE = 2**8  # Batch size for training.
EPOCHS = 10
LATENT_DIM = 100  # Latent dimensionality of the encoding space.

GLOVE_DIM = 50

WEIGHTS_PATH = RESOURCES_DIR.joinpath('weights')


def replace_with_embs(sents, word_embedding):
    """
    Converts sentences to lists of their word embeddings.

    sents: list of tokenized sentences - each sentence is a list of strings

    unknown_vec: object of sent_emb.algorithms.unknown.Unknown abstract class

    returns: list of sentences
        sentence: list of embeddings
        embedding: list of floats
    """

    word_vec_dict = word_embedding.embeddings(sents)

    sents_vec = []
    for sent in sents:
        cur_sent = []
        for word in sent:
            cur_sent.append(word_vec_dict[word])
        sents_vec.append(cur_sent)

    return sents_vec


def get_random_subsequence(sequence, result_size):
    """
    Computes random subsequence of size 'result_size' of python list 'sequence'.
    """
    seq_len = len(sequence)
    assert result_size <= seq_len

    selected_indices = np.sort(np.random.permutation(seq_len)[: result_size])

    return [sequence[ind] for ind in np.nditer(selected_indices)]


def align_sents(sents_vec, padding_vec, cut_rate=0.8):
    """
    Fits each sentence to has equal number of words (dependent on 'cut_rate').

    sents_vec: list of sentences of vectorized words
               (see return type of replace_with_embs() function)

    padding_vec: np.array of type np.float and length GLOVE_DIM
                 is used when there is not enough words in the sentence.

    cut_rate: coefficient of [0; 1] interval
              Target number of words per sentence (num_encoder_words) is set to be the minimal
              integer such that at least 'cut_rate' fraction of original sentences are of length
              less or equal 'num_encoder_words'.

    returns: list of sentences (in format as 'sents_vec')
             each sentence consists of MAX_ENCODER_WORDS words.
    """
    assert 0 <= cut_rate <= 1

    sent_lengths = sorted([len(sent) for sent in sents_vec])
    num_encoder_words = sent_lengths[int(math.ceil(cut_rate * len(sent_lengths)))]

    for i in range(len(sents_vec)):
        if len(sents_vec[i]) <= num_encoder_words:
            sents_vec[i].extend([padding_vec for _ in range(num_encoder_words - len(sents_vec[i]))])
        else:
            sents_vec[i] = get_random_subsequence(sents_vec[i], num_encoder_words)
        assert len(sents_vec[i]) == num_encoder_words

    return sents_vec


def preprocess_sents(sents, word_embedding):
    """
    Prepares sentences to be put into Seq2Seq neural net.

    sents: list of tokenized sentences - each sentence is a list of strings

    returns: numpy 3-D array of floats, which represents list of sentences of vectorized words
    """

    sents_vec = replace_with_embs(sents, word_embedding)

    padding_vec = np.zeros(GLOVE_DIM, dtype=np.float)
    aligned_sents = align_sents(sents_vec, padding_vec)

    return np.array(aligned_sents, dtype=np.float)


class Seq2Seq(BaseAlgorithm):
    def __init__(self, name='s2s_gru_sts1215_g50', force_load=True):
        """
        Constructs Seq2Seq model and optionally loads saved state of the model from disk.

        name: short details of model - it's used as a prefix of name of file with saved model.

        force_load: if True and there aren't proper files with saved model, then __init__
                    print error message and terminates execution of script.
        """

        self.name = name
        self.all_weights_path = WEIGHTS_PATH.joinpath('{}.h5'.format(name))
        self.encoder_weights_path = WEIGHTS_PATH.joinpath('{}_enc.h5'.format(name))

        self.force_load = force_load

        self.word_embedding = GloVeSmall()

        config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, GLOVE_DIM))
        encoder = GRU(LATENT_DIM, return_state=True)
        encoder_outputs, state_h = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h]
        self.encoderModel = Model(encoder_inputs, encoder_states)

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, GLOVE_DIM))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_gru = GRU(LATENT_DIM, return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_gru(decoder_inputs,
                                         initial_state=encoder_states)
        decoder_dense = Dense(GLOVE_DIM, activation='linear')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.model.compile(optimizer='rmsprop', loss='mean_squared_error')

        # Try to load saved model from disk.
        if self.all_weights_path.exists() and self.encoder_weights_path.exists():
            print('Loading weights from files:\n{}\n{}'.format(str(self.all_weights_path),
                                                               str(self.encoder_weights_path)))
            self.model.load_weights(str(self.all_weights_path))
            self.encoderModel.load_weights(str(self.encoder_weights_path))
        else:
            if self.force_load:
                error_msg = \
'''ERROR: Weights not found and force_load==True
       If you really want to create new files with weights, please add
       --alg-kwargs='{"force_load": false}' as a param of the script.
'''
                print(error_msg)
                assert False
            else:
                print('Weights not found - model will be created from scratch.')

    def fit(self, sents):
        """
        We don't want to train model during evaluation - it would take too much time.
        """
        pass

    def improve_weights(self, sents, epochs=EPOCHS):
        """
        Real training of the model.

        sents: list of tokenized sentences - each sentence is a list of strings
        """
        sents_vec = preprocess_sents(sents, self.word_embedding)
        print("Shape of sentences after preprocessing:", sents_vec.shape)

        encoder_input_data = sents_vec

        # i-th cell of decoder receives as input a word embedding,
        # which (i-1)-th cell of encoder received as input.
        decoder_input_data = np.zeros(sents_vec.shape, dtype='float32')
        decoder_input_data[:, 1:, :] = sents_vec[:, :-1, :]

        decoder_target_data = np.copy(decoder_input_data)

        self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                       batch_size=BATCH_SIZE, epochs=epochs)

        # Save model
        WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(self.all_weights_path))
        self.encoderModel.save_weights(str(self.encoder_weights_path))

    def transform(self, sents):
        sents_vec = preprocess_sents(sents, self.word_embedding)
        print("Shape of sentences after preprocessing:", sents_vec.shape)
        assert sents_vec.shape[0] == len(sents) and sents_vec.shape[2] == GLOVE_DIM

        embs = self.encoderModel.predict(sents_vec)

        assert embs.shape == (sents_vec.shape[0], LATENT_DIM)

        return embs

    def get_resources(self, dataset):
        self.word_embedding.get_resources(dataset)


def improve_model(algorithm, tokenizer):
    """
    Runs training of Seq2Seq 'algorithm' model on STS16 training set.
    """

    algorithm.get_resources(STS(tokenizer))

    print('Reading training set...')
    sents = flatten_sent_pairs(read_train_set(16, tokenizer))
    print('...done.')

    algorithm.improve_weights(sents)
