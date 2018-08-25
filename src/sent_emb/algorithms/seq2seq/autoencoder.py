import numpy as np

from keras import Input, Model
from keras.layers import GRU, Dense, Masking, LSTM
from keras.regularizers import l1_l2

from sent_emb.algorithms.glove_utility import GloVe, GloVeSmall
from sent_emb.algorithms.seq2seq.preprocessing import preprocess_sent_pairs
from sent_emb.algorithms.seq2seq.utility import (Seq2Seq, load_model_weights, save_model_weights, get_words)


BATCH_SIZE = 2**9  # Batch size for training.


def define_models(word_emb_dim, latent_dim, words):
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

    decoder_gru = GRU(latent_dim, return_sequences=True,
                      return_state=True, recurrent_regularizer=l1_l2(0.00, 0.0005))
    decoder_outputs, _ = decoder_gru(decoder_inputs,
                                     initial_state=encoder_states)
    decoder_dense = Dense(word_emb_dim, activation='linear', name='embeddings')
    decoder_outputs = decoder_dense(decoder_outputs)


    # Surprisingly, linear activation with mean_squared_error loss work best
    # Possibly 0.0002 is too high
    decoder_bow = Dense(words, activation='linear',
                        kernel_regularizer=l1_l2(0.00, 0.001),
                        name='BOW')(encoder_outputs)


    # Define complete model, which will be trained later.
    complete_model = Model([encoder_inputs, decoder_inputs], [decoder_outputs, decoder_bow])
    complete_model.summary()

    return complete_model, encoder_model


def prepare_models(name, word_emb_dim, latent_dim, words, force_load=True):
    complete_model, encoder_model = define_models(word_emb_dim, latent_dim, words)

    load_model_weights(name, complete_model, encoder_model, force_load=force_load)

    return complete_model, encoder_model


class Autoencoder(Seq2Seq):
    """
    This algorithm uses autoencoding neural net based on seq2seq architecture.
    """

    def __init__(self, name='s2s_autoencoder', force_load=True, glove_dim=50, latent_dim=100, flip=True, word_frac=0.9):
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
        self.latent_dim = latent_dim
        self.force_load = force_load
        self.flip = flip

        self.vectorizer = None
        self.word_frac = word_frac

        self.encoder_model = None
        self.complete_model = None
        # self.build_model()

        # self._check_members_presence()

    def build_model(self, words):
        print("build...")
        self.complete_model, self.encoder_model = \
            prepare_models(self.name, self.word_embedding.get_dim(), self.latent_dim,
                           words=words, force_load=self.force_load)

        self.complete_model.compile(optimizer='rmsprop', loss='mean_squared_error',
                                    loss_weights=[0.1, 1.])


    def improve_weights(self, sent_pairs, epochs, **kwargs):
        first_sents_vec, second_sents_vec = preprocess_sent_pairs(sent_pairs, self.word_embedding)
        first_sents = [' '.join(gs.sent1) for gs in sent_pairs]

        # only for STS data
        # sents_vec = np.concatenate([first_sents_vec, second_sents_vec])
        sents_vec = first_sents_vec

        if self.complete_model is None:
            print("Building model...")
            self.vectorizer = get_words(sent_pairs, self.word_frac)
            words_count = len(self.vectorizer.vocabulary_)
            self.build_model(words_count)
            print("...done")

        bow_target_data = self.vectorizer.transform(first_sents).todense()

        print("Shape of sentences after preprocessing:", sents_vec.shape)

        encoder_input_data = sents_vec

        # i-th cell of decoder receives as input a word embedding,
        # which (i-1)-th cell of encoder received as input.
        decoder_input_data = np.zeros(sents_vec.shape, dtype='float32')
        decoder_input_data[:, 1:, :] = sents_vec[:, :-1, :]

        decoder_target_data = np.copy(decoder_input_data)

        self.complete_model.fit(x=[encoder_input_data, decoder_input_data], y=[decoder_target_data, bow_target_data],
                                batch_size=BATCH_SIZE, epochs=epochs)

        save_model_weights(self.name, self.complete_model, self.encoder_model)
