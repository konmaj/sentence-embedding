import numpy as np
from keras.layers import Dense, BatchNormalization
from keras import Input, Model

from sklearn.feature_extraction.text import CountVectorizer

from sent_emb.evaluation.model import BaseAlgorithm
from sent_emb.algorithms.unknown import UnknownVector
from sent_emb.algorithms.glove_utility import GloVe


GLOVE_DIM = 300


def parse_sents(sents, word_embedding, maxlen):
    result = np.zeros((len(sents), maxlen, GLOVE_DIM), dtype=np.float)

    word_to_vec = word_embedding.embeddings(sents)

    for ids, sent in enumerate(sents):
        for idw, word in enumerate(sent):
            result[ids, idw] = word_to_vec[word]

    result = result.reshape((result.shape[0], result.shape[1]*result.shape[2]))

    return result


def build_model(target_dim, input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(500, activation='relu')(inputs)
    x = BatchNormalization()(x)
    outputs = Dense(target_dim, activation='relu')(x)
    encoder = Model(inputs=inputs, outputs=outputs)

    inputs = Input(shape=(input_dim,))
    x = Dense(500, activation='relu')(encoder(inputs))
    outputs = Dense(output_dim, activation='relu')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return encoder, model


class SimpleAutoencoder(BaseAlgorithm):
    def __init__(self, unknown=UnknownVector(GLOVE_DIM), maxlen=100, target_dim=300):
        '''
            unknown: handler of words not appearing in GloVe
            maxlen: maximal sentence length
            target_dim: dimension of encoding space
        '''
        self.encoder = None
        self.model = None
        self.word_embedding = GloVe(unknown)
        self.maxlen = maxlen
        self.target_dim = target_dim

    def get_resources(self, task):
        self.word_embedding.get_resources(task)

    def fit(self, sents):
        x_train = parse_sents(sents, self.word_embedding, self.maxlen)

        vectorizer = CountVectorizer(preprocessor=(lambda line: ' '.join(line).lower()), binary=True)
        y_train = vectorizer.fit_transform(sents).toarray()

        self.encoder, self.model = build_model(self.target_dim, self.maxlen * GLOVE_DIM, y_train.shape[1])

        self.model.fit(x_train, y_train, epochs=1, batch_size=64, shuffle=True)

    def transform(self, sents):
        x_test = parse_sents(sents, self.word_embedding, self.maxlen)

        return self.encoder.predict(x_test)
