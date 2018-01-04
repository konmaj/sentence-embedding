import numpy as np
from pathlib import Path

from sent_emb.algorithms.unkown import UnknownVector
from sent_emb.algorithms.glove_utility import GLOVE_LINES, GLOVE_FILE, read_file

GLOVE_DIM = 300

def embeddings(sents, unknown=UnknownVector(GLOVE_DIM)):
    '''
        sents: numpy array of sentences to compute embeddings

        returns: numpy 2-D array of embeddings;
        length of single embedding is arbitrary, but have to be
        consistent across the whole result
    '''
    where = {}
    words = set()

    result = np.zeros((sents.shape[0], GLOVE_DIM), dtype=np.float)
    count = np.zeros((sents.shape[0], 1))

    for idx, sent in enumerate(sents):
        for word in sent:
            if word != '':
                if word not in where:
                    where[word] = []
                where[word].append(idx)

    def process(word, vec, _):
        words.add(word)
        unknown.see(word, vec)
        if word in where:
            for idx in where[word]:
                result[idx] += vec
                count[idx][0] += 1
                
    read_file(GLOVE_FILE, process, should_count=True)
    
    for word in where:
        if word not in words:
            for idx in where[word]:
                result[idx] += unknown.get(word)
                count[idx][0] += 1
            print('Not found word \'' + word + '\'')

    result /= count

    return result

def test_run():
    embeddings(np.array(['This is a very good sentence, my friend.',
                        'Birdie lives in the United States.',
                        '\"He is not a suspect anymore.\" John said.']))
