import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, GRU, Dot

from sent_emb.algorithms.glove_utility import GloVeSmall
from sent_emb.algorithms.seq2seq.preprocessing import preprocess_sent_pairs
from sent_emb.algorithms.seq2seq.utility import (Seq2Seq, load_model_weights, save_model_weights)
from sent_emb.evaluation.model import get_gold_standards


BATCH_SIZE = 2 ** 8  # Batch size for training.
EPOCHS = 5
LATENT_DIM = 100  # Latent dimensionality of the encoding space.


def define_models(word_emb_dim, latent_dim):
    K.set_learning_phase(1)

    # Define the encoder.
    encoder_inputs = [Input(shape=(None, word_emb_dim), name='encoder_input_sent{}'.format(i))
                      for i in range(2)]
    encoder_gru = GRU(latent_dim, return_state=True, name='encoder_GRU')

    # Get encoder hidden states - sentence embeddings.
    encoder_states_h = []
    for i in range(2):
        _, state_tmp = encoder_gru(encoder_inputs[i])
        encoder_states_h.append(state_tmp)

    encoder_model = Model(encoder_inputs[0], encoder_states_h[0])

    # Control cosine similarity of trained embeddings.
    gs_outputs = Dot(axes=1, normalize=True, name='cosine_similarity')(encoder_states_h)

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = [Input(shape=(None, word_emb_dim), name='decoder_input_sent{}'.format(i))
                      for i in range(2)]
    decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True, name='decoder_GRU')

    decoder_gru_outputs = []
    for i in range(2):
        decoder_gru_output, _ = decoder_gru(decoder_inputs[i],
                                            initial_state=encoder_states_h[i])
        decoder_gru_outputs.append(decoder_gru_output)

    decoder_dense = Dense(word_emb_dim, activation='linear', name='decoder_dense')
    decoder_outputs = [decoder_dense(decoder_gru_outputs[i]) for i in range(2)]

    # Define complete model, which will be trained later.
    complete_model = Model(encoder_inputs + decoder_inputs, decoder_outputs + [gs_outputs])
    complete_model.summary()

    return complete_model, encoder_model


def prepare_models(name, word_emb_dim, latent_dim, force_load=True):
    complete_model, encoder_model = define_models(word_emb_dim, latent_dim)

    load_model_weights(name, complete_model, encoder_model, force_load=force_load)

    return complete_model, encoder_model


class AutoencoderWithCosine(Seq2Seq):
    """
    This algorithm uses autoencoding neural net based on seq2seq architecture.

    Apart from autoencoding properties, the model pays attention also to gold standard scores
    of similarity between pairs of sentences.
    """

    def __init__(self, name='s2s_gru_cos_g50_sts1215', force_load=True):
        """
        Constructs Seq2Seq model and optionally loads saved state of the model from disk.

        name: short details of model - it's used as a prefix of name of file with saved model.

        force_load: if True and there aren't proper files with saved model, then __init__
                    prints error message and terminates execution of the script.
        """
        super(AutoencoderWithCosine, self).__init__(GloVeSmall(), LATENT_DIM)

        self.name = name
        self.force_load = force_load

        self.complete_model, self.encoder_model = \
            prepare_models(name, self.word_embedding.get_dim(), LATENT_DIM,
                           force_load=force_load)

        self.complete_model.compile(optimizer='rmsprop', loss='mean_squared_error')

        self._check_members_presence()

    def improve_weights(self, sent_pairs, epochs=EPOCHS):
        # Filter out pairs, which do not have gold standard scores.
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

        self.complete_model.fit(encoder_input_data + decoder_input_data,
                                decoder_target_data + [np.array(gs_target_data)],
                                batch_size=BATCH_SIZE, epochs=epochs)

        save_model_weights(self.name, self.complete_model, self.encoder_model)
