import numpy as np

from keras import Input, Model
from keras.layers import GRU, Dense, Masking
from keras.regularizers import l1_l2

from sent_emb.algorithms.glove_utility import GloVe, GloVeSmall
from sent_emb.algorithms.seq2seq.preprocessing import preprocess_sent_pairs
from sent_emb.algorithms.seq2seq.utility import (Seq2Seq, load_model_weights, save_model_weights)


BATCH_SIZE = 2**8  # Batch size for training.


def define_models(word_emb_dim, latent_dim):

    # Define the encoder
    encoder_inputs = Input(shape=(None, word_emb_dim))
    encoder_mask = Masking()
    encoder = GRU(latent_dim, return_state=True, recurrent_regularizer=l1_l2(0.00, 0.001))
    encoder_outputs, state_h = encoder(encoder_mask(encoder_inputs))

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h]
    encoder_model = Model(encoder_inputs, encoder_states)

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, word_emb_dim))

    decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True, recurrent_regularizer=l1_l2(0.00, 0.001))
    decoder_outputs, _ = decoder_gru(decoder_inputs,
                                     initial_state=encoder_states)
    decoder_dense = Dense(word_emb_dim, activation='linear')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define complete model, which will be trained later.
    complete_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    complete_model.summary()

    return complete_model, encoder_model


def prepare_models(name, word_emb_dim, latent_dim, force_load=True):
    complete_model, encoder_model = define_models(word_emb_dim, latent_dim)

    load_model_weights(name, complete_model, encoder_model, force_load=force_load)

    return complete_model, encoder_model


class Autoencoder(Seq2Seq):
    """
    This algorithm uses autoencoding neural net based on seq2seq architecture.
    """

    def __init__(self, name='s2s_autoencoder', force_load=True, glove_dim=50, latent_dim=100, flip=True):
        """
        Constructs Seq2Seq model and optionally loads saved state of the model from disk.

        name: short details of model - it's used as a prefix of name of file with saved model.

        force_load: if True and there aren't proper files with saved model, then __init__
                    prints error message and terminates execution of the script.

        latent_dim: latent dimensionality of the encoding space.
        """
        if glove_dim == 50:
            super(Autoencoder, self).__init__(GloVeSmall(), latent_dim)
        else:
            super(Autoencoder, self).__init__(GloVe(), latent_dim)

        self.name = name
        self.force_load = force_load
        self.flip = flip

        self.complete_model, self.encoder_model = \
            prepare_models(name, self.word_embedding.get_dim(), latent_dim,
                           force_load=force_load)

        self.complete_model.compile(optimizer='rmsprop', loss='mean_squared_error')

        self._check_members_presence()

    def improve_weights(self, sent_pairs, epochs, **kwargs):
        first_sents_vec, second_sents_vec = preprocess_sent_pairs(sent_pairs, self.word_embedding)

        sents_vec = np.concatenate([first_sents_vec, second_sents_vec])

        print("Shape of sentences after preprocessing:", sents_vec.shape)

        encoder_input_data = sents_vec

        # i-th cell of decoder receives as input a word embedding,
        # which (i-1)-th cell of encoder received as input.
        decoder_input_data = np.zeros(sents_vec.shape, dtype='float32')
        decoder_input_data[:, 1:, :] = sents_vec[:, :-1, :]

        decoder_target_data = np.copy(decoder_input_data)

        # if self.flip:
        #     encoder_input_data = np.flip(encoder_input_data, axis=1)

        self.complete_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                batch_size=BATCH_SIZE, epochs=epochs)

        save_model_weights(self.name, self.complete_model, self.encoder_model)
