import numpy as np
from shutil import copyfile

from sent_emb.algorithms.path_utility import EMBEDDINGS_DIR
from sent_emb.downloader.downloader import zip_download_and_extract
from sent_emb.evaluation.model import WordEmbedding, DataSet
from sent_emb.algorithms.unknown import UnknownVector

GLOVE_DIR = EMBEDDINGS_DIR.joinpath('glove')
RAW_GLOVE_FILE_300 = GLOVE_DIR.joinpath('glove.840B.300d.txt')
RAW_GLOVE_FILE_50 = GLOVE_DIR.joinpath('glove.6B.50d.txt')
GLOVE_FILE = GLOVE_DIR.joinpath('glove_cropped.txt')
GLOVE_DIM = 300


def get_zip_file(glove_file):
    if glove_file == RAW_GLOVE_FILE_300:
        return glove_file.stem + '.zip'
    else:
        return glove_file.with_suffix('').stem + '.zip'


def get_glove_file(glove_file, name):
    """
    :param glove_file: name of the glove file from which we cropped
    :param name: name of the tokenizer
    :return: name of the cropped glove file for the given tokenizer
    """
    return GLOVE_DIR.joinpath(glove_file.stem + '_cropped_' + name + '.txt')


def read_file(file_path, f, should_count=False, discard=0):
    """
    :param file_path: Name of the glove file, normally just use GLOVE_FILE
    :param f: f(word, vec, raw_line) callback for reading the file
        param vec: np.array of size GLOVE_DIM with word embedding
    :param should_count: whether we should print diagnostic info every 100k lines
    :param discard: how many first lines should be ignored
        it's usually 0, but sometimes there are some meta data in the first line (eg. FastText)
    """
    line_count = 0
    glove_file = open(file_path)
    for idx, raw_line in enumerate(glove_file):
        if idx >= discard:
            line = raw_line.split(' ')
            word = line[0]
            data = line[1:]
            # dealing with trailing spaces in files
            while data[-1] == "" or data[-1] == '\n':
                data = data[:-1]
            vec = np.array(data, dtype=np.float)
            f(word, vec, raw_line)
            if should_count:
                line_count += 1
                if line_count % (100 * 1000) == 0:
                    print('  line_count: ', line_count)


def create_glove_subset(dataset, glove_file, name):
    """
    Checks whether cropped GloVe file exists and
    crops GloVe file to contain only words used in task (if
    :param dataset: dataset data object
    :param glove_file: glove_file to be cropped
    :param name: name to be append to the glove_file (usually tokenizer name)
    """
    if get_glove_file(glove_file, name).exists():
        print('Cropped GloVe file already exists')
    else:
        file = open(get_glove_file(glove_file, name), 'w')

        def crop(word, _, line):
            if word in dataset.word_set:
                file.write(line)

        print('Cropping GloVe set - it may take a while...')
        read_file(glove_file, crop, should_count=True)
        file.close()

    copyfile(get_glove_file(glove_file, name), GLOVE_FILE)


def download_glove(glove_file):
    base_url = "http://nlp.stanford.edu/data/"

    zip_file = get_zip_file(glove_file)
    print("Check GloVe data...")
    if glove_file.exists():
        print("... Embeddings", glove_file, "found")
    else:
        print("Downloading", glove_file, "...")
        zip_download_and_extract(base_url + zip_file, GLOVE_DIR)
        print("...", glove_file, "downloaded")


def get_glove_resources(dataset, glove_file):
    assert isinstance(dataset, DataSet)
    download_glove(glove_file)
    create_glove_subset(dataset, glove_file, dataset.tokenizer_name())


class GloVe(WordEmbedding):
    def __init__(self, unknown, glove_file=RAW_GLOVE_FILE_300, dim=300):
        self.unknown = unknown
        self.glove_file = glove_file
        self.dim = dim

    def get_resources(self, dataset):
        get_glove_resources(dataset, self.glove_file)

    def embeddings(self, sents):
        words = set()
        for sent in sents:
            for word in sent:
                words.add(word)

        result = {}

        def process(w, vec, _):
            self.unknown.see(w, vec)
            result[w] = vec

        read_file(GLOVE_FILE, process)

        for word in words:
            if word not in result:
                result[word] = self.unknown.get(word)

        return result

    def get_dim(self):
        return self.dim


class GloVeSmall(GloVe):
    def __init__(self, unknown=UnknownVector(50)):
        super(GloVeSmall, self).__init__(unknown, glove_file=RAW_GLOVE_FILE_50, dim=50)
