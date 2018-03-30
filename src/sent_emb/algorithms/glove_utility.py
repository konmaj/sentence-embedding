import numpy as np
from shutil import copyfile

from sent_emb.algorithms.path_utility import EMBEDDINGS_DIR

GLOVE_DIR = EMBEDDINGS_DIR.joinpath('glove')
RAW_GLOVE_FILE_300 = GLOVE_DIR.joinpath('glove.840B.300d.txt')
RAW_GLOVE_FILE_50 = GLOVE_DIR.joinpath('glove.6B.50d.txt')
GLOVE_FILE = GLOVE_DIR.joinpath('glove_cropped.txt')
GLOVE_DIM = 300


def get_glove_file(glove_file, name):
    '''
    :param glove_file: name of the glove file from which we cropped
    :param name: name of the tokenizer
    :return: name of the cropped glove file for the given tokenizer
    '''
    return GLOVE_DIR.joinpath(glove_file.stem + '_cropped_' + name + '.txt')


def read_file(file_path, f, should_count=False):
    '''
    :param file_path: Name of the glove file, normally just use GLOVE_FILE
    :param f(word, vec, raw_line): callback for reading the file
        :param vec: np.array of size GLOVE_DIM with word embedding
    :param should_count: whether we should print diagnostic info every 100k lines
    '''
    line_count = 0
    glove_file = open(file_path)
    for raw_line in glove_file:
        line = raw_line[:-1].split(' ')
        word = line[0]
        vec = np.array(line[1:], dtype=np.float)
        f(word, vec, raw_line)
        if should_count:
            line_count += 1
            if line_count % (100 * 1000) == 0:
                print('  line_count: ', line_count)


def create_glove_subset(task, glove_file, name):
    '''
    Checks whether cropped GloVe file exists and
    crops GloVe file to contain only words used in task (if
    :param evaluation: evaluation object for this task
    :param glove_file: glove_file to be cropped
    :param name: name to be append to the glove_file (usually tokenizer name)
    '''
    if get_glove_file(glove_file, name).exists():
        print('Cropped GloVe file already exists')
    else:
        file = open(get_glove_file(glove_file, name), 'w')

        def crop(word, _, line):
            if word in task.word_set():
                file.write(line)

        print('Cropping GloVe set...')
        read_file(glove_file, crop, should_count=True)
        file.close()

    copyfile(get_glove_file(glove_file, name), GLOVE_FILE)


def get_glove_resources(task, glove_file):
    create_glove_subset(task, glove_file, task.tokenizer_name())


class GloVe:
    def __init__(self, unknown, glove_file=RAW_GLOVE_FILE_300, dim=300):
        self.unknown = unknown
        self.glove_file = glove_file
        self.dim = dim

    def get_resources(self, task):
        get_glove_resources(task, self.glove_file)

    def embeddings(self, words):
        result = {}

        def process(word, vec, _):
            self.unknown.see(word, vec)
            result[word] = vec

        read_file(GLOVE_FILE, process)

        for word in words:
            if word not in result:
                result[word] = self.unknown.get(word)

        return result

    def get_dim(self):
        return self.dim


def gloVe_small(unknown):
    return GloVe(unknown, glove_file=RAW_GLOVE_FILE_50, dim=50)