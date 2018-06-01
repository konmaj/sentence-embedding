import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, GRU, Dot
from keras.regularizers import l1

from sent_emb.algorithms.glove_utility import GloVe, GloVeSmall
from sent_emb.algorithms.seq2seq.preprocessing import preprocess_sent_pairs
from sent_emb.algorithms.seq2seq.utility import (Seq2Seq, load_model_weights, save_model_weights)
from sent_emb.evaluation.model import get_gold_standards


BATCH_SIZE = 2 ** 8  # Batch size for training.


def define_models(word_emb_dim, latent_dim, reg_coef=None, dropout=0.0, recurrent_dropout=0.0,
                  recurrent_activation='hard_sigmoid'):
    K.set_learning_phase(1)

    # Define the encoder.
    encoder_inputs = [Input(shape=(None, word_emb_dim), name='encoder_input_sent{}'.format(i))
                      for i in range(2)]

    regularizer = None if reg_coef is None else l1(reg_coef)
    encoder_gru = GRU(latent_dim, return_state=True, name='encoder_GRU',
                      recurrent_regularizer=regularizer,
                      dropout=dropout, recurrent_dropout=recurrent_dropout,
                      recurrent_activation=recurrent_activation)

    # Get encoder hidden states - sentence embeddings.
    encoder_states_h = []
    for i in range(2):
        _, state_tmp = encoder_gru(encoder_inputs[i])
        encoder_states_h.append(state_tmp)

    encoder_model = Model(encoder_inputs[0], encoder_states_h[0])

    # Control cosine similarity of trained embeddings.
    gs_outputs = Dot(axes=1, normalize=True, name='cosine_similarity')(encoder_states_h)

    # Define complete model, which will be trained later.
    complete_model = Model(encoder_inputs, [gs_outputs])
    complete_model.summary()

    return complete_model, encoder_model


def prepare_models(name, word_emb_dim, latent_dim, force_load=True, **kwargs):
    complete_model, encoder_model = define_models(word_emb_dim, latent_dim, **kwargs)

    load_model_weights(name, complete_model, encoder_model, force_load=force_load)

    return complete_model, encoder_model


class Cosine(Seq2Seq):
    """
    This algorithm uses autoencoding neural net based on seq2seq architecture.

    Apart from autoencoding properties, the model pays attention also to gold standard scores
    of similarity between pairs of sentences.
    """

    def __init__(self, name='s2s_cos_g50_sts1215_d100', force_load=True, latent_dim=100,
                 reg_coef=None, loss='mean_squared_error', optimizer='rmsprop',
                 dropout=0.0, recurrent_dropout=0.0, glove_dim=50,
                 recurrent_activation='hard_sigmoid'):
        """
        Constructs Seq2Seq model and optionally loads saved state of the model from disk.

        name: short details of model - it's used as a prefix of name of file with saved model.

        force_load: if True and there aren't proper files with saved model, then __init__
                    prints error message and terminates execution of the script.

        latent_dim: latent dimensionality of the encoding space.
        """
        if glove_dim == 50:
            super().__init__(GloVeSmall(), latent_dim)
        elif glove_dim == 300:
            super().__init__(GloVe(), latent_dim)
        else:
            assert False

        self.name = name
        self.force_load = force_load

        self.complete_model, self.encoder_model = \
            prepare_models(name, self.word_embedding.get_dim(), latent_dim,
                           force_load=force_load, reg_coef=reg_coef,
                           dropout=dropout, recurrent_dropout=recurrent_dropout,
                           recurrent_activation=recurrent_activation)

        self.complete_model.compile(optimizer=optimizer, loss=loss)

        self._check_members_presence()

    def improve_weights(self, sent_pairs, epochs, **kwargs):
        # Filter out pairs, which do not have gold standard scores.
        sent_pairs = [pair for pair in sent_pairs if pair.gs is not None]

        first_sents_vec, second_sents_vec = preprocess_sent_pairs(sent_pairs, self.word_embedding)
        print("Shape of sentences after preprocessing:", first_sents_vec.shape, second_sents_vec.shape)

        encoder_input_data = [first_sents_vec, second_sents_vec]

        gs_target_data = np.array(get_gold_standards(sent_pairs)) * (2 / 5) - 1

        self.complete_model.fit(encoder_input_data, [np.array(gs_target_data)],
                                batch_size=BATCH_SIZE, epochs=epochs)

        save_model_weights(self.name, self.complete_model, self.encoder_model)
