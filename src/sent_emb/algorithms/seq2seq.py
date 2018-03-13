import numpy as np
from pathlib import Path
import sys
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense

from sent_emb.algorithms.glove_utility import read_file
from sent_emb.algorithms.unkown import UnknownVector
from sent_emb.evaluation.model import BaseAlgorithm
from sent_emb.evaluation import sts

BATCH_SIZE = 64  # Batch size for training.
LATENT_DIM = 100  # Latent dimensionality of the encoding space.
MAX_ENCODER_WORDS = 100

GLOVE_DIM = 50
GLOVE_50D_FILE = Path('/', 'opt', 'resources', 'embeddings', 'glove', 'glove.6B.50d.txt')
if not GLOVE_50D_FILE.exists():
    error_msg = \
'''ERROR: file {0} doesn't exist
        You can download missing file from http://nlp.stanford.edu/data/glove.6B.zip
        This will be automized soon.'''
    print(error_msg)
    sys.exit(1)

WEIGHTS_PATH = Path('/', 'opt', 'resources', 'weights')
WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)


def replace_with_embs(sents, unknown_vec):
    '''
    Converts sentences to lists of their word embeddings.

    sents: the same as in Seq2Seq.improve_weights() method

    unknown_vec: object of sent_emb.algorithms.unknown.Unknown abstract class

    returns: list of sentences
        sentence: list of embeddings
        embedding: list of floats
    '''
    word_set = set()
    for sent in sents:
        word_set.update(sent)

    word_embs = {}
    unknown_vec = UnknownVector(GLOVE_DIM)

    def capture_word_embs(word, vec, _):
        if word in word_set:
            word_embs[word] = vec
            unknown_vec.see(word, vec)
    print('Reading file:', str(GLOVE_50D_FILE))
    read_file(str(GLOVE_50D_FILE), capture_word_embs, should_count=True)

    sents_vec = []
    for sent in sents:
        cur_sent = []
        for word in sent:
            cur_sent.append(word_embs[word] if word in word_embs else unknown_vec.get(None))
        sents_vec.append(cur_sent)

    return sents_vec, unknown_vec


def align_sents(sents_vec, unknown_vec):
    '''
    Fits each sentence to has number of words == MAX_ENCODER_WORDS

    sents_vec: list of sentences of vectorized words
               (see return type of replace_with_embs() function)

    unknown_vec: object of sent_emb.algorithms.unknown.Unknown abstract class

    returns: list of sentences (in format as 'sents_vec')
             each sentence consists of MAX_ENCODER_WORDS words.
    '''
    for sent in sents_vec:
        assert len(sent) <= MAX_ENCODER_WORDS

        sent.extend([unknown_vec.get(None) for _ in range(MAX_ENCODER_WORDS - len(sent))])

        assert len(sent) == MAX_ENCODER_WORDS
    return sents_vec


def preprocess_sents(sents):
    '''
    Prepares sentences to be put into Seq2Seq neural net.

    sents: the same as in Seq2Seq.improve_weights() method

    returns: numpy 3-D array of floats, which represents list of sentences of vectorized words
    '''

    sents_vec, unknown_vec = replace_with_embs(sents, UnknownVector(GLOVE_DIM))

    aligned_sents = align_sents(sents_vec, unknown_vec)

    return np.array(aligned_sents)


class Seq2Seq(BaseAlgorithm):
    def __init__(self, name='s2s_lstm_sts1215_g50', force_load=True):
        '''
        Constructs Seq2Seq model and optionally loads saved state of the model from disk.

        name: short details of model - it's used as a prefix of name of file with saved model.

        force_load: if True and there aren't proper files with saved model, then __init__
                    print error message and terminates execution of script.
        '''
        self.name = name
        self.all_weights_path = WEIGHTS_PATH.joinpath('{}.h5'.format(name))
        self.encoder_weights_path = WEIGHTS_PATH.joinpath('{}_enc.h5'.format(name))

        self.force_load = force_load

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, GLOVE_DIM))
        encoder = LSTM(LATENT_DIM, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]
        self.encoderModel = Model(encoder_inputs, encoder_states)

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, GLOVE_DIM))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
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
       If you really want to create new files with weights, please add keyword argument
       force_load==false using --alg-kwargs param of the script.
'''
                print(error_msg)
                assert False
            else:
                print('Weights not found - model will be created from scratch.')

    def fit(self, sents):
        '''
        We don't want to train model during evaluation - it would take too much time.
        '''
        pass

    def improve_weights(self, sents, epochs=1):
        '''
        Real training of the model.

        sents: numpy array of tokenized sentences
               (shaped as in sent_emb.evaluation.sts.read_sts_input())
        '''
        sents_vec = preprocess_sents(sents)

        encoder_input_data = sents_vec

        # i-th cell of decoder receives as input a word embedding,
        # which (i-1)-th cell of encoder received as input.
        decoder_input_data = np.zeros(sents_vec.shape, dtype='float32')
        decoder_input_data[:, 1:, :] = sents_vec[:, :-1, :]

        decoder_target_data = np.copy(decoder_input_data)

        self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                       batch_size=BATCH_SIZE, epochs=epochs)

        # Save model
        self.model.save_weights(str(self.all_weights_path))
        self.encoderModel.save_weights(str(self.encoder_weights_path))

    def transform(self, sents):
        sents_vec = preprocess_sents(sents)
        assert sents_vec.shape == (sents.shape[0], MAX_ENCODER_WORDS, GLOVE_DIM)

        embs = self.encoderModel.predict(sents_vec)

        assert len(embs) == 2

        return np.concatenate(embs, axis=1)


def improve_model(algorithm, tokenizer):
    '''
    Runs training of Seq2Seq 'algorithm' model on STS16 traing set.
    '''
    print('Reading training set...')
    sents = sts.read_train_set(16, tokenizer)
    print('...done.')
    algorithm.improve_weights(sents)
