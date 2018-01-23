import numpy as np
from pathlib import Path

DOWNLOAD_DIR = Path('/', 'opt', 'resources')
GLOVE_DIR = DOWNLOAD_DIR.joinpath('embeddings', 'glove')
RAW_GLOVE_FILE = GLOVE_DIR.joinpath('glove.840B.300d.txt')
RAW_GLOVE_LINES = 2196017
GLOVE_FILE = GLOVE_DIR.joinpath('glove_cropped.txt')
GLOVE_DIM = 300


def glove_file(name):
    return GLOVE_DIR.joinpath('glove_cropped_' + name + '.txt')


def read_file(file_path, f, should_count = False):
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


def create_glove_subset(word_set, name):
    file = open(glove_file(name), 'w')
    def crop(word, _, line):
        if word in word_set:
            file.write(line)
    print('Cropping GloVe set...')
    print('  Lines overall: ', RAW_GLOVE_LINES)
    read_file(RAW_GLOVE_FILE, crop, should_count=True)
    file.close()
