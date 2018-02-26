import numpy as np
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
from keras import backend as K, Input, Model

from sklearn.feature_extraction.text import CountVectorizer

from sent_emb.algorithms.unkown import UnknownVector
from sent_emb.algorithms.glove_utility import GLOVE_DIM, GLOVE_FILE, read_file

model = None
encoder = None
default_maxlen = 80
default_target_dim = 300


def parse_sents(sents, unknown=UnknownVector(GLOVE_DIM), maxlen=default_maxlen):
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

    return result, result.shape[1]


def train_model(sents, unknown=UnknownVector(GLOVE_DIM), maxlen=default_maxlen, target_dim=default_target_dim):
    global model
    global encoder

    x_train, input_dim = parse_sents(sents, unknown, maxlen)

    vectorizer = CountVectorizer(preprocessor=(lambda line: ' '.join(line).lower()))
    y_train = vectorizer.fit_transform(sents).toarray()

    K.set_learning_phase(1)

    inputs = Input(shape=(input_dim,))
    x = Dense(500, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(target_dim, activation='relu')(x)
    # x = BatchNormalization()(x)
    encoder = Model(inputs=inputs, outputs=x)

    inputs = Input(shape=(input_dim,))
    # x = BatchNormalization()(encoder)
    x = Dense(500, activation='relu')(encoder(inputs))
    outputs = Dense(y_train.shape[1], activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, batch_size=64, shuffle=True)


def get_single_embedding(sentences):
    global model
    global encoder

    return encoder.predict([sentences])


def embeddings(sents, unknown=UnknownVector(GLOVE_DIM), maxlen=default_maxlen, target_dim=default_target_dim):
    '''
        sents: numpy array of sentences to compute embeddings
        unknown: handler of words not appearing in GloVe
        maxlen: maximal sentence length
        target_dim: the embedding dimension

        returns: numpy 2-D array of embeddings;
        length of single embedding is equal to target_dim
    '''

    x_test, dim = parse_sents(sents, unknown, maxlen)

    result = get_single_embedding(x_test)

    return result
