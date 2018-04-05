import numpy as np
from os import system
from shutil import copyfile

from sent_emb.algorithms.path_utility import EMBEDDINGS_DIR, OTHER_RESOURCES_DIR
from sent_emb.downloader.downloader import mkdir_if_not_exist, zip_download_and_extract
from sent_emb.evaluation.model import WordEmbedding
from sent_emb.algorithms.unknown import UnknownVector, NoUnknown

FASTTEXT_DIR = EMBEDDINGS_DIR.joinpath('fasttext')

FASTTEXT_FILE = FASTTEXT_DIR.joinpath('wiki.en.vec')
FASTTEXT_BIN = FASTTEXT_DIR.joinpath('wiki.en.bin')
FASTTEXT_CROPPED = FASTTEXT_DIR.joinpath('cropped.txt')
FASTTEXT_UNKNOWN = FASTTEXT_DIR.joinpath('unknown_answers.txt')

FASTTEXT_GITHUB_DIR = OTHER_RESOURCES_DIR.joinpath('fasttext').joinpath('fastText-0.1.0')


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


def get_answers(name):
    question = get_unknown_file(name)
    answer = get_answers_file(name)

    if not answer.exists():
        print('Computing FastText vectors for unknown words...')
        system(FASTTEXT_GITHUB_DIR.as_posix() + '/fasttext print-word-vectors '
               + FASTTEXT_BIN.as_posix()
               + ' < ' + question.as_posix() + ' > ' + answer.as_posix())
        print('...vectors computed')
    else:
        print('FastText vectors for unknown words found')


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
        create_fasttext_subset(task.word_set, name)

    get_answers(name)

    copyfile(get_cropped_file(name), FASTTEXT_CROPPED)
    copyfile(get_answers_file(name), FASTTEXT_UNKNOWN)


def get_fasttext_resources(task):
    print('Checking for fastText')
    EMB_URL = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip'
    GIT_URL = 'https://github.com/facebookresearch/fastText/archive/v0.1.0.zip'

    mkdir_if_not_exist(EMBEDDINGS_DIR)
    if mkdir_if_not_exist(FASTTEXT_DIR):
        print('FastText embeddings not found')
        zip_download_and_extract(EMB_URL, FASTTEXT_DIR)
    else:
        print('Found fastText embeddings')

    mkdir_if_not_exist(OTHER_RESOURCES_DIR)
    if mkdir_if_not_exist(FASTTEXT_GITHUB_DIR.parent):
        print('FastText github not found')
        zip_download_and_extract(GIT_URL, FASTTEXT_GITHUB_DIR.parent)
    else:
        print('Found fasttext github')
    system('cd ' + FASTTEXT_GITHUB_DIR.as_posix() + ' && make')

    fasttext_preprocessing(task, task.tokenizer_name())


def normalize(vec):
    return vec / np.linalg.norm(vec)

class FastText(WordEmbedding):
    def get_dim(self):
        return 300

    def get_resources(self, task):
        get_fasttext_resources(task)

    def embeddings(self, sents):
        words = set()
        for sent in sents:
            for word in sent:
                words.add(word)
        answer = {}

        def process(word, vec, _):
            if word in words:
                answer[word] = normalize(vec)

        read_file(FASTTEXT_CROPPED, process)
        read_file(FASTTEXT_UNKNOWN, process)
        return answer


class FastTextWithoutUnknown(WordEmbedding):
    def get_dim(self):
        return 300

    def get_resources(self, task):
        get_fasttext_resources(task)

    def embeddings(self, sents):
        words = set()
        for sent in sents:
            for word in sent:
                words.add(word)

        answer = {}
        unknown = UnknownVector(300)

        def process(word, vec, _):
            unknown.see(word, vec)
            if word in words:
                answer[word] = vec

        read_file(FASTTEXT_CROPPED, process)

        for word in words:
            if word not in answer:
                answer[word] = unknown.get(word)

        return answer
