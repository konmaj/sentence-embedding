import numpy as np
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
from keras import backend as K, Input, Model

from sklearn.feature_extraction.text import CountVectorizer

from sent_emb.evaluation.model import BaseAlgorithm
from sent_emb.algorithms.unknown import UnknownVector
from sent_emb.algorithms.glove_utility import GLOVE_DIM, GLOVE_FILE, read_file


def parse_sents(sents, unknown, maxlen):
    where = {}
    words = set()

    result = np.zeros((sents.shape[0], maxlen, GLOVE_DIM), dtype=np.float)

    for ids, sent in enumerate(sents):
        for idw, word in enumerate(sent):
            if word != '':
                if word not in where:
                    where[word] = []
                where[word].append((ids, idw))

    def process(word, vec, _):
        words.add(word)
        unknown.see(word, vec)
        if word in where:
            for (ids, idw) in where[word]:
                result[ids, idw] = vec

    read_file(GLOVE_FILE, process)

    result = result.reshape((result.shape[0], result.shape[1]*result.shape[2]))

    return result


def buildModel(target_dim, input_dim, output_dim):
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
        self.unknown = unknown
        self.maxlen = maxlen
        self.target_dim = target_dim

    def get_resources(self, _):
        pass

    def fit(self, sents):
        x_train = parse_sents(sents, self.unknown, self.maxlen)

        vectorizer = CountVectorizer(preprocessor=(lambda line: ' '.join(line).lower()), binary=True)
        y_train = vectorizer.fit_transform(sents).toarray()

        self.encoder, self.model = buildModel(self.target_dim, self.maxlen * GLOVE_DIM, y_train.shape[1])

        self.model.fit(x_train, y_train, epochs=1, batch_size=64, shuffle=True)

    def transform(self, sents):
        x_test = parse_sents(sents, self.unknown, self.maxlen)

        return self.encoder.predict(x_test)
