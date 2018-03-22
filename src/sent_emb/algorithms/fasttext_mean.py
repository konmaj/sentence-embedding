import numpy as np

from sent_emb.downloader.downloader import get_fasttext
from sent_emb.algorithms.glove_utility import GLOVE_DIM
from sent_emb.algorithms.path_utility import EMBEDDINGS_DIR
from sent_emb.algorithms.unknown import Unknown

FASTTEXT_DIR = EMBEDDINGS_DIR.joinpath('fasttext')
FASTTEXT_FILE = FASTTEXT_DIR.joinpath('wiki-news-300d-1M-subword.vec')


def read_file(file_path, f, should_count=False):
    line_count = 0
    glove_file = open(file_path)
    header = ""
    for raw_line in glove_file:
        if header == "":
            header = raw_line
        else:
            line = raw_line[:-2].split(' ')
            word = line[0]
            vec = np.array(line[1:], dtype=np.float)
            f(word, vec, raw_line)
            if should_count:
                line_count += 1
                if line_count % (100 * 1000) == 0:
                    print('  line_count: ', line_count)


def normalize(vec):
    return vec / np.linalg.norm(vec)


class FastText(Unknown):
    def __init__(self):
        Unknown.__init__(self)
        self.mem = {}
        self.sum = np.array([])
        self.count = 0

    def see(self, w, vec):
        self.mem[w] = vec
        if len(self.sum) == 0:
            self.sum = vec
        else:
            self.sum += vec
        self.count += 1

    def get(self, word):
        if word not in self.mem:
            return self.sum / self.count
        return self.mem[word]


def embeddings(sents, word_embedding=FastText()):
    get_fasttext()

    result = np.zeros((sents.shape[0], GLOVE_DIM), dtype=np.float)
    count = np.zeros((sents.shape[0], 1))

    def process(word, vec, _):
        word_embedding.see(word, vec)

    read_file(FASTTEXT_FILE, process, should_count=True)

    for idx, sent in enumerate(sents):
        for word in sent:
            if word != '':
                result[idx] += normalize(word_embedding.get(word))
                count[idx][0] += 1

    result /= count

    return result
