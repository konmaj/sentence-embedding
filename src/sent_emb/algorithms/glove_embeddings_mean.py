import numpy as np
from pathlib import Path

from sent_emb.algorithms.glove_utility import GLOVE_LINES, GLOVE_FILE, read_file

def embeddings(sents):
    '''
        sents: numpy array of sentences to compute embeddings

        returns: numpy 2-D array of embeddings;
        length of single embedding is arbitrary, but have to be
        consistent across the whole result
    '''
    where = {}
    words = set()

    result = np.zeros((sents.shape[0], 300), dtype=np.float)
    count = np.zeros((sents.shape[0], 1))

    for idx, sent in enumerate(sents):
        chars = [',', ':', '/', '(', ')', '?', '!', '.', '"', "$", '“', '”', '#', ';', '%']

        for c in chars:
            sent = sent.replace(c, ' ' + c + ' ')

        sent = sent.split(' ')
        for word in sent:
            if word != '':
                if word not in where:
                    where[word] = []
                where[word].append(idx)

    def process(word, vec, _):
        words.add(word)
        if word in where:
            for idx in where[word]:
                result[idx] += vec
                count[idx][0] += 1
    print('Reading GloVe...')
    print('  Lines overall: ' + str(GLOVE_LINES))
    read_file(GLOVE_FILE, process, should_count=True)
    result /= count

    for word in where:
        if not word in words:
            print('Not found word \'' + word + '\'')

    return result

def test_run():
    embeddings(np.array(['This is a very good sentence, my friend.',
                        'Birdie lives in the United States.',
                        '\"He is not a suspect anymore.\" John said.']))
