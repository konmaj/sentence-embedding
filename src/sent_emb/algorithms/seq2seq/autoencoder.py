import numpy as np

from keras import Input, Model
from keras.layers import GRU, Dense, Masking, LSTM, Subtract, Lambda
from keras.regularizers import l1_l2
from keras import backend as K

from sent_emb.algorithms.glove_utility import GloVe, GloVeSmall
from sent_emb.algorithms.seq2seq.preprocessing import preprocess_sent_pairs
from sent_emb.algorithms.seq2seq.utility import (Seq2Seq, load_model_weights, save_model_weights, get_words)
from sent_emb.evaluation.model import SentPairWithGs

BATCH_SIZE = 2**9  # Batch size for training.


def emb_loss(y_true, y_pred):
    pos = K.sum(K.square(y_true * (1.-y_pred)), axis=-1) / (1 + K.sum(y_true, axis=-1))
    neg = K.sum(K.square((1.-y_true) * y_pred), axis=-1) / (1 + K.sum(1.-y_true, axis=-1))
    return (2*pos+neg)/3


def define_models(word_emb_dim, latent_dim, words, embeddings):
    # Define the encoder
    # encoder_inputs = [Input(shape=(None, word_emb_dim)) for _ in range(2)]
    encoder_inputs = Input(shape=(None, word_emb_dim))
    encoder_mask = Masking()
    encoder_bot = GRU(latent_dim, return_sequences=True, recurrent_regularizer=l1_l2(0.00, 0.00))
    encoder = GRU(latent_dim, return_state=True, recurrent_regularizer=l1_l2(0.00, 0.00))

    encoder_outputs, state_h   = encoder(encoder_bot(encoder_mask(encoder_inputs)))
    # _,               state_h_2 = encoder(encoder_mask(encoder_inputs[1]))
    # state_h_2 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(state_h_2)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h]
    # encoder_model = Model(encoder_inputs[0], encoder_states)
    encoder_model = Model(encoder_inputs, encoder_states)


    # # Set up the decoder, using `encoder_states` as initial state.
    # decoder_inputs = Input(shape=(None, word_emb_dim))
    #
    # decoder_gru = GRU(latent_dim, return_sequences=True,
    #                   return_state=True, recurrent_regularizer=l1_l2(0.00, 0.001))
    # decoder_outputs, _ = decoder_gru(decoder_inputs,
    #                                  initial_state=encoder_states)
    # decoder_dense = Dense(word_emb_dim, activation='linear', name='embeddings')
    # decoder_outputs = decoder_dense(decoder_outputs)


    # Surprisingly, linear activation with mean_squared_error loss work best
    decoder_bow = Dense(len(words), activation='linear',
                        kernel_regularizer=l1_l2(0.00, 0.000),
                        # trainable=False,
                        name='BOW')(encoder_outputs)


    # decoder_prev = Dense(words, activation='linear',
    #                      kernel_regularizer=l1_l2(0.00, 0.001),
    #                      name='prev')(encoder_outputs)
    #
    #
    # decoder_next = Dense(words, activation='linear',
    #                      kernel_regularizer=l1_l2(0.00, 0.001),
    #                      name='next')(encoder_outputs)


    # decoder_next_emb = Dense(word_emb_dim, activation='linear',
    #                          kernel_regularizer=l1_l2(0.00, 0.001),
    #                          name='next_embedding')(encoder_outputs)
    # next_emb_diff = Subtract()([decoder_next_emb, state_h])


    # Define complete model, which will be trained later.
    complete_model = Model(encoder_inputs, [decoder_bow])
    complete_model.summary()

    # print(complete_model.layers[-1].get_weights()[0].shape)
    # # complete_model.layers[-1].set_weights(weights)
    # w = [[]] * len(words)
    # k = [[word] for word in words.keys()]
    # print('k', len(k))
    #
    # print('emb', len(embeddings.embeddings(k)))
    #
    # for it in embeddings.embeddings(k).items():
    #     try:
    #         # print(it[0])
    #         # print(it[1])
    #         # print(words[it[0]])
    #         w[words[it[0]]] = it[1]
    #     except KeyError:
    #         continue
    #
    # print(np.array(w).shape)
    # print(np.array(w))
    #
    # complete_model.layers[-1].set_weights([np.transpose(np.array(w)), np.zeros(len(words))])

    return complete_model, encoder_model


def prepare_models(name, word_emb_dim, latent_dim, words, embeddings, force_load=True):
    complete_model, encoder_model = define_models(word_emb_dim, latent_dim, words, embeddings)

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
                           words=words, embeddings=self.word_embedding,
                           force_load=self.force_load)

        self.complete_model.compile(optimizer='rmsprop',
                                    # loss='mean_squared_error'
                                    loss=emb_loss,
                                    # loss_weights=[1., 2., 1.]
                                    )


    def improve_weights(self, sent_pairs, epochs, **kwargs):
        # print(sent_pairs[:5])
        # sent_pairs.sort(key=lambda s: len(s.sent1))
        # print(sent_pairs[:5])

        sent_pairs_normal = [SentPairWithGs(gs.sent1, gs.sent2, gs.gs) for gs in sent_pairs]
        first_sents_vec, second_sents_vec = preprocess_sent_pairs(sent_pairs_normal, self.word_embedding, rand=0.00)
        first_sents  = [' '.join(gs.sent1) for gs in sent_pairs]
        # second_sents = [' '.join(gs.sent2[0]) for gs in sent_pairs]
        # third_sents  = [' '.join(gs.sent2[1]) for gs in sent_pairs]

        # only for STS data
        # sents_vec = np.concatenate([first_sents_vec, second_sents_vec])
        sents_vec = first_sents_vec

        if self.complete_model is None:
            print("Building model...")
            self.vectorizer = get_words(sent_pairs, self.word_frac)
            words = self.vectorizer.vocabulary_
            self.build_model(words)
            print("...done")

        bow_target_data = self.vectorizer.transform(first_sents).todense()
        # prev_target_data = self.vectorizer.transform(second_sents).todense()
        # next_target_data = self.vectorizer.transform(third_sents).todense()
        # next_emb_data = self.encoder_model.predict(x=[second_sents_vec], batch_size=BATCH_SIZE)
        next_emb_diff_data = np.zeros((sents_vec.shape[0], sents_vec.shape[2]))

        print("Shape of sentences after preprocessing:", sents_vec.shape)

        encoder_input_data = sents_vec

        # i-th cell of decoder receives as input a word embedding,
        # which (i-1)-th cell of encoder received as input.
        decoder_input_data = np.zeros(sents_vec.shape, dtype='float32')
        decoder_input_data[:, 1:, :] = sents_vec[:, :-1, :]

        decoder_target_data = np.copy(decoder_input_data)

        # encoder_input_data += np.random.rand(
        #     encoder_input_data.shape[0],
        #     encoder_input_data.shape[1],
        #     encoder_input_data.shape[2]) * 0.02

        self.complete_model.fit(x=[encoder_input_data], y=[bow_target_data],
                                shuffle=True,
                                batch_size=BATCH_SIZE, epochs=epochs)

        save_model_weights(self.name, self.complete_model, self.encoder_model)
