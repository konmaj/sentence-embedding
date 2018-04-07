import keras
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import GRU, Dense

from sent_emb.algorithms.glove_utility import GloVeSmall
from sent_emb.algorithms.seq2seq.seq2seq import Seq2Seq, WEIGHTS_PATH, GLOVE_DIM, LATENT_DIM, EPOCHS, \
    preprocess_sent_pairs, BATCH_SIZE, preprocess_sents


class Autoencoder(Seq2Seq):
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

    def improve_weights(self, sent_pairs, epochs=EPOCHS):
        first_sents_vec, second_sents_vec = preprocess_sent_pairs(sent_pairs, self.word_embedding)
        sents_vec = np.concatenate([first_sents_vec, second_sents_vec])

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
