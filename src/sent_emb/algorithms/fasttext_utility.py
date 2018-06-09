import numpy as np
from os import system
from shutil import copyfile

from sent_emb.algorithms.path_utility import EMBEDDINGS_DIR, OTHER_RESOURCES_DIR
from sent_emb.downloader.downloader import mkdir_if_not_exist, zip_download_and_extract
from sent_emb.evaluation.model import WordEmbedding
from sent_emb.algorithms.unknown import UnknownVector
from sent_emb.algorithms.glove_utility import read_file, GloVe

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


def get_answers(name):
    question = get_unknown_file(name)
    answer = get_answers_file(name)

    if not answer.exists():
        print('Computing FastText vectors for unknown words...')
        command = FASTTEXT_GITHUB_DIR.as_posix() + '/fasttext print-word-vectors '\
                  + FASTTEXT_BIN.as_posix()\
                  + ' < ' + question.as_posix() + ' > ' + answer.as_posix()
        msg = '''
        WARNING: This computation may cause thrashing if:
        1) You have less than 11 GB of RAM and
        2) you didn't set memory limits for Docker container.
        See doc/README_docker.md for further info.
        '''

        print(msg)
        system(command)
        print('...vectors computed')
    else:
        print('FastText vectors for unknown words found')


def create_fasttext_subset(word_set, name):
    """
    Creates files with subsets of words for fasttext (unknown and known)
    :param word_set: set of words used in task
    :param name: name of the tokenizer
    """
    seen = set()

    file = open(get_cropped_file(name), 'w')

    def crop(w, _, line):
        if w in word_set:
            seen.add(w)
            file.write(line)

    read_file(FASTTEXT_FILE, crop, should_count=True, discard=1)
    file.close()

    unknown = open(get_unknown_file(name), 'w')
    for word in word_set:
        if word not in seen:
            unknown.write(word + '\n')
    unknown.close()


def fasttext_preprocessing(dataset, name):
    if get_unknown_file(name).exists() and get_cropped_file(name).exists():
        print('Cropped Fasttext file exists')
    else:
        print('Creating Fasttext cropped file')
        create_fasttext_subset(dataset.word_set, name)

    get_answers(name)

    copyfile(get_cropped_file(name), FASTTEXT_CROPPED)
    copyfile(get_answers_file(name), FASTTEXT_UNKNOWN)


def get_fasttext_resources(dataset):
    print('Checking for fastText')
    emb_url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip'
    git_url = 'https://github.com/facebookresearch/fastText/archive/v0.1.0.zip'

    mkdir_if_not_exist(EMBEDDINGS_DIR)
    if mkdir_if_not_exist(FASTTEXT_DIR):
        print('FastText embeddings not found')
        zip_download_and_extract(emb_url, FASTTEXT_DIR)
    else:
        print('Found fastText embeddings')

    mkdir_if_not_exist(OTHER_RESOURCES_DIR)
    if mkdir_if_not_exist(FASTTEXT_GITHUB_DIR.parent):
        print('FastText github not found')
        zip_download_and_extract(git_url, FASTTEXT_GITHUB_DIR.parent)
    else:
        print('Found fasttext github')
    system('cd ' + FASTTEXT_GITHUB_DIR.as_posix() + ' && make')

    fasttext_preprocessing(dataset, dataset.tokenizer_name())


def normalize(vec):
    return vec / np.linalg.norm(vec)


class FastText(WordEmbedding):
    @staticmethod
    def get_dim():
        return 300

    @staticmethod
    def get_resources(dataset):
        get_fasttext_resources(dataset)

    @staticmethod
    def embeddings(sents):
        words = set()
        for sent in sents:
            for word in sent:
                words.add(word)
        answer = {}

        def process(w, vec, _):
            if w in words:
                answer[w] = vec

        read_file(FASTTEXT_CROPPED, process)

        sum_len = 0
        count = 0
        for w in answer:
            vec = answer[w]
            sum_len += np.linalg.norm(vec)
            count += 1

        def process_unknown(w, vec, _):
            if w in words:
                answer[w] = sum_len / count * normalize(vec)
        read_file(FASTTEXT_UNKNOWN, process_unknown)
        return answer


class FastTextWithGloVeLength(FastText):
    def __init__(self, glove=GloVe(unknown=UnknownVector(300))):
        self.glove = glove
        assert(glove.get_dim() == 300)
        super(FastTextWithGloVeLength, self).__init__()

    def get_resources(self, dataset):
        super(FastTextWithGloVeLength, self).get_resources(dataset)
        self.glove.get_resources(dataset)

    def embeddings(self, sents):
        glove_emb = self.glove.embeddings(sents)
        fasttext_emb = super(FastTextWithGloVeLength, self).embeddings(sents)
        for word in fasttext_emb:
            v = fasttext_emb[word]
            fasttext_emb[word] = v / np.linalg.norm(v) * np.linalg.norm(glove_emb[word])
        return fasttext_emb


class FastTextNormalized(FastText):
    @staticmethod
    def embeddings(sents):
        fasttext_emb = FastText.embeddings(sents)
        for word in fasttext_emb:
            v = fasttext_emb[word]
            fasttext_emb[word] = v / np.linalg.norm(v)
        return fasttext_emb


class FastTextWithoutUnknown(WordEmbedding):
    @staticmethod
    def get_dim():
        return 300

    @staticmethod
    def get_resources(dataset):
        get_fasttext_resources(dataset)

    @staticmethod
    def embeddings(sents):
        words = set()
        for sent in sents:
            for word in sent:
                words.add(word)

        answer = {}
        unknown = UnknownVector(300)

        def process(w, vec, _):
            unknown.see(w, vec)
            if w in words:
                answer[w] = vec

        read_file(FASTTEXT_CROPPED, process)

        for word in words:
            if word not in answer:
                answer[word] = unknown.get(word)

        return answer
