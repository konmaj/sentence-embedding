import numpy as np
from pathlib import Path
from shutil import copyfile

from sent_emb.algorithms.path_utility import EMBEDDINGS_DIR
from sent_emb.downloader.downloader import mkdir_if_not_exist, zip_download_and_extract

FASTTEXT_DIR = EMBEDDINGS_DIR.joinpath('fasttext')

FASTTEXT_FILE = FASTTEXT_DIR.joinpath('wiki-news-300d-1M-subword.vec')
FASTTEXT_CROPPED = FASTTEXT_DIR.joinpath('cropped.txt')
FASTTEXT_UNKNOWN = FASTTEXT_DIR.joinpath('unknown_answers.txt')


def get_unknown_file(name):
    return FASTTEXT_DIR.joinpath('unknown_' + name + '.txt')


def get_answers_file(name):
    return FASTTEXT_DIR.joinpath('answers_' + name + '.txt')


def get_cropped_file(name):
    return FASTTEXT_DIR.joinpath('fasttext_cropped_' + name + '.txt')


def read_file(file_path, f, should_count=False, discard=0):
    '''
    :param file_path: Name of the fasttext file, normally use FASTTEXT_FILE
    :param f(word, vec, raw_line): callback for reading the file
        :param vec: np.array with word embedding
    :param should_count: whether we should print diagnostic info every 100k lines
    :param discard: how many first lines should be ignored
        for downloaded data from the Internet it's usually 1, for self-constructed files 0
    '''
    line_count = 0
    file = open(file_path)
    for idx, raw_line in enumerate(file):
        if idx >= discard:
            line = raw_line.split(' ')
            word = line[0]
            vec = np.array(line[1:-1], dtype=np.float)
            f(word, vec, raw_line)
            if should_count:
                line_count += 1
                if line_count % (100 * 1000) == 0:
                    print('  line_count: ', line_count)


def create_fasttext_subset(word_set, name):
    '''
    Creates files with subsets of words for fasttext (unknown and known)
    :param word_set: set of words used in task
    :param name: name of the tokenizer
    '''
    seen = set()

    file = open(get_cropped_file(name), 'w')

    def crop(word, _, line):
        if word in word_set:
            seen.add(word)
            file.write(line)

    read_file(FASTTEXT_FILE, crop, should_count=True, discard=1)
    file.close()

    unknown = open(get_unknown_file(name), 'w')
    for word in word_set:
        if word not in seen:
            unknown.write(word + '\n')
    unknown.close()


def fasttext_preprocessing(task, name):
    if get_unknown_file(name).exists() and get_cropped_file(name).exists():
        print('Cropped Fasttext file exists')
    else:
        print('Creating Fasttext cropped file')
        create_fasttext_subset(task.all_words(), name)

    copyfile(get_cropped_file(name), FASTTEXT_CROPPED)
    copyfile(get_answers_file(name), FASTTEXT_UNKNOWN)


def get_fasttext_resources(task):
    print('Checking for fastText')
    URL = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip'

    path_emb = EMBEDDINGS_DIR
    mkdir_if_not_exist(path_emb)
    path = path_emb.joinpath('fasttext')
    if mkdir_if_not_exist(path):
        print('FastText not found')
        zip_download_and_extract(URL, path)
    else:
        print('Found fastText')
    fasttext_preprocessing(task, task.tokenizer_name())
