import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Dense, GRU, Dot

import tensorflow as tf

from sent_emb.algorithms.seq2seq.utility import (WEIGHTS_PATH, Seq2Seq, preprocess_sent_pairs,
                                                 preprocess_sents)
from sent_emb.algorithms.glove_utility import GloVeSmall
from sent_emb.evaluation.model import get_gold_standards


BATCH_SIZE = 2 ** 8  # Batch size for training.
EPOCHS = 10
LATENT_DIM = 100  # Latent dimensionality of the encoding space.


class AutoencoderWithCosine(Seq2Seq):
    def __init__(self, name='s2s_gru_sts1215_g50_tmp', force_load=True):
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

        self.word_embedding = GloVeSmall()
        self.glove_dim = self.word_embedding.get_dim()

        config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)

        # Define an input sequence and process it.
        encoder_inputs = [Input(shape=(None, self.glove_dim), name='encoder_input_sent{}'.format(i)) for i in range(2)]
        encoder_gru = GRU(LATENT_DIM, return_state=True, name='encoder_GRU')
        encoder_states_h = []
        for i in range(2):
            _, state_tmp = encoder_gru(encoder_inputs[i])
            encoder_states_h.append(state_tmp)

        # We discard `encoder_outputs` and only keep the states.
        self.encoderModel = Model(encoder_inputs[0], encoder_states_h[0])

        gs_outputs = Dot(axes=1, normalize=True, name='cosine_similarity')(encoder_states_h)

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = [Input(shape=(None, self.glove_dim), name='decoder_input_sent{}'.format(i)) for i in range(2)]
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_gru = GRU(LATENT_DIM, return_sequences=True, return_state=True, name='decoder_GRU')

        decoder_gru_outputs = []
        for i in range(2):
            decoder_gru_output, _ = decoder_gru(decoder_inputs[i],
                                                initial_state=encoder_states_h[i])
            decoder_gru_outputs.append(decoder_gru_output)

        decoder_dense = Dense(self.glove_dim, activation='linear', name='decoder_dense')
        decoder_outputs = [decoder_dense(decoder_gru_outputs[i]) for i in range(2)]

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model(encoder_inputs + decoder_inputs, decoder_outputs + [gs_outputs])

        self.model.compile(optimizer='rmsprop', loss='mean_squared_error')

        self.model.summary()

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
        '''
        Real training of the model.

        sents: numpy array of tokenized sentences
               (shaped as in sent_emb.evaluation.sts.read_sts_input())
        '''
        sent_pairs = [pair for pair in sent_pairs if pair.gs is not None]

        first_sents_vec, second_sents_vec = preprocess_sent_pairs(sent_pairs, self.word_embedding)
        print("Shape of sentences after preprocessing:", first_sents_vec.shape, second_sents_vec.shape)

        encoder_input_data = [first_sents_vec, second_sents_vec]

        decoder_input_data = []
        for i in range(2):
            # i-th cell of decoder receives as input a word embedding,
            # which (i-1)-th cell of encoder received as input.
            decoder_input_data.append(np.zeros(encoder_input_data[i].shape, dtype='float32'))
            decoder_input_data[-1][:, 1:, :] = encoder_input_data[i][:, :-1, :]

        decoder_target_data = [np.copy(decoder_input_data[i]) for i in range(2)]

        gs_target_data = np.array(get_gold_standards(sent_pairs)) * (2 / 5) - 1

        self.model.fit(encoder_input_data + decoder_input_data, decoder_target_data + [np.array(gs_target_data)],
                       batch_size=BATCH_SIZE, epochs=epochs)

        # Save model
        WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(self.all_weights_path))
        self.encoderModel.save_weights(str(self.encoder_weights_path))

    def transform(self, sents):
        sents_vec = preprocess_sents(sents, self.word_embedding)
        print("Shape of sentences after preprocessing:", sents_vec.shape)
        assert sents_vec.shape[0] == sents.shape[0] and sents_vec.shape[2] == self.glove_dim

        embs = self.encoderModel.predict(sents_vec)

        assert embs.shape == (sents_vec.shape[0], LATENT_DIM)

        return embs
