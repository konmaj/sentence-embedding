import csv
import datetime
import subprocess

import numpy as np
from pathlib import Path
from nltk.tokenize import word_tokenize

from sent_emb.algorithms.glove_utility import create_glove_subset, GLOVE_FILE
from sent_emb.downloader.downloader import mkdir_if_not_exist

TEST_NAMES = {
    12: ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews'],
    13: ['headlines', 'OnWN', 'FNWN'],
    14: ['deft-forum', 'deft-news', 'headlines', 'images', 'OnWN', 'tweet-news'],
    15: ['answers-forums', 'answers-students', 'belief', 'headlines', 'images'],
    16: ['answer-answer', 'headlines', 'plagiarism', 'postediting', 'question-question'],
}

DATASETS_PATH = Path('/', 'opt', 'resources', 'datasets')
LOG_PATH = Path('/', 'opt', 'resources', 'log')

# Script from STS16 seems to be backward compatible with file formats from former years.
GRADING_SCRIPT_PATH = DATASETS_PATH.joinpath('STS16', 'data', 'correlation-noconfidence.pl')


def vector_len(vec):
    return np.sum(np.square(vec)) ** 0.5


def cos(vec0, vec1):
    dot_product = np.sum(vec0 * vec1)
    return dot_product / (vector_len(vec0) * vector_len(vec1))


def compute_similarity(emb_pairs):
    result = []

    for pair in emb_pairs:
        sim = cos(pair[0], pair[1])
        result.append((sim + 1) * 5 / 2) # scale interval from [-1; 1] to [0; 5]

    return np.array(result)


def generate_similarity_file(emb_func, input_path, output_path):

    # read test data
    sents = []

    with open(input_path, 'r') as test_file:
        test_reader = csv.reader(test_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in test_reader:
            assert len(row) == 2 \
                or len(row) == 4 # STS16 contains also source of each sentence
            sents.extend(row[:2])
    sents = np.array([word_tokenize(s) for s in sents])

    # compute embeddings
    embs = emb_func(sents)
    assert len(embs) == len(sents)

    # generate similarities between pairs of sentences
    embs.shape = (embs.shape[0] // 2, 2, embs.shape[1])
    similarities = compute_similarity(embs)

    # write file with similarities
    with open(output_path, 'w+') as out_file:
        for sim in similarities:
            out_file.write('{}\n'.format(sim))


def eval_sts_year(emb_func, year):
    assert year in TEST_NAMES

    sts_name = 'STS{}'.format(year)
    print('Evaluating on datasets from {}'.format(sts_name))

    sts_dir = DATASETS_PATH.joinpath(sts_name)
    data_dir = sts_dir.joinpath('data')
    out_dir = sts_dir.joinpath('out')

    in_prefix = 'STS.input'
    out_prefix = 'STS.output'
    gs_prefix = 'STS.gs'

    mkdir_if_not_exist(LOG_PATH)
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    log_file_name = '{}-{}.txt'.format(sts_name, cur_time)
    log_file_path = LOG_PATH.joinpath(log_file_name)

    for test_name in TEST_NAMES[year]:
        in_name = '{}.{}.txt'.format(in_prefix, test_name)
        out_name = '{}.{}.txt'.format(out_prefix, test_name)
        gs_name = '{}.{}.txt'.format(gs_prefix, test_name)

        print('Evaluating on file: {}'.format(in_name))

        # generate out
        in_path = data_dir.joinpath(in_name)
        out_path = out_dir.joinpath(out_name)
        gs_path = data_dir.joinpath(gs_name)

        generate_similarity_file(emb_func, in_path, out_path)

        # compare out with gold standard
        script = GRADING_SCRIPT_PATH

        score = subprocess.check_output(
            ['perl', script, gs_path, out_path],
            universal_newlines=True,
        )

        log_msg = 'Test name: {}\n{}\n'.format(test_name, score)
        with open(log_file_path, 'a+') as log_file:
            log_file.write(log_msg)
        print(log_msg)

def create_glove_sts_subset():
    #Create set of words which appeared in STS files
    sts_words = {'house', 'Bird', 'play', 'mother'}
    if GLOVE_FILE.exists():
        print('Cropped GloVe file already exists')
    else:
        create_glove_subset(sts_words)

def eval_sts_all(emb_func):
    create_glove_sts_subset()
    for year in TEST_NAMES:
        eval_sts_year(emb_func, year)

