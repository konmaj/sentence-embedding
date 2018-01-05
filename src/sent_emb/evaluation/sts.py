import csv
import datetime
import subprocess
import re

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


def get_sts_path(year):
    assert year in TEST_NAMES
    return DATASETS_PATH.joinpath('STS{}'.format(year))


def get_sts_input_path(year, test_name):
    assert year in TEST_NAMES

    sts_path = get_sts_path(year)
    input_name = 'STS.input.{}.txt'.format(test_name)

    return sts_path.joinpath('data', input_name)


def get_sts_gs_path(year, test_name):
    assert year in TEST_NAMES

    sts_path = get_sts_path(year)
    gs_name = 'STS.gs.{}.txt'.format(test_name)

    return sts_path.joinpath('data', gs_name)


def get_sts_output_path(year, test_name):
    assert year in TEST_NAMES

    sts_path = get_sts_path(year)
    output_name = 'STS.output.{}.txt'.format(test_name)

    return sts_path.joinpath('out', output_name)


def get_cur_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def read_sts_input(file_path):
    sents = []
    with open(file_path, 'r') as test_file:
        test_reader = csv.reader(test_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in test_reader:
            assert len(row) == 2 \
                or len(row) == 4 # STS16 contains also source of each sentence
            sents.extend(row[:2])
            
    return np.array([word_tokenize(s) for s in sents])


def generate_similarity_file(emb_func, input_path, output_path):
    '''
    Runs given embedding function ('emb_func') on a single STS task (without
    computing score).

    Writes output in format described in section Output Files of file
    resources/datasets/STS16/data/README.txt
    '''
    # read test data
    sents = read_sts_input(input_path)

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


def get_grad_script_res(output):
    res = re.search(r'^Pearson: (\d\.\d{5})$', output)
    assert res is not None
    return float(res.groups()[0]) # throws exception in case of wrong conversion


def eval_sts_year(emb_func, year, year_file=False):
    '''
    Evaluates given embedding function on STS inputs from given year.

    If year_file=True, generates file with results in the LOG_PATH directory.

    Returns list of "Pearson's r * 100" of each input
    (ordered as in TEST_NAMES[year]).
    '''
    assert year in TEST_NAMES

    print('Evaluating on datasets from STS{}'.format(year))
    results = []

    mkdir_if_not_exist(LOG_PATH)
    cur_time = get_cur_time_str()
    log_file_name = 'STS{}-{}.txt'.format(year, cur_time)
    log_file_path = LOG_PATH.joinpath(log_file_name)

    for test_name in TEST_NAMES[year]:

        print('Evaluating on test {} from STS{}'.format(test_name, year))

        # generate out
        in_path = get_sts_input_path(year, test_name)
        out_path = get_sts_output_path(year, test_name)
        gs_path = get_sts_gs_path(year, test_name)

        generate_similarity_file(emb_func, in_path, out_path)

        # compare out with gold standard
        script = GRADING_SCRIPT_PATH
        output = subprocess.check_output(
            ['perl', script, gs_path, out_path],
            universal_newlines=True,
        )
        score = get_grad_script_res(output) * 100

        results.append(score)

        # update log file
        log_msg = 'Test name: {}\n100*Pearson: {:7.3f}\n'.format(test_name, score)
        print(log_msg)
        if year_file:
            with open(log_file_path, 'a+') as log_file:
                log_file.write(log_msg)

    return results


def create_glove_sts_subset():
    '''
    1) Computes set of words which appeared in STS input files.
    2) Creates reduced GloVe file, which contains only words that appeared
       in STS.
    '''
    if GLOVE_FILE.exists():
        print('Cropped GloVe file already exists')
        return

    sts_words = set()
    for year, test_names in TEST_NAMES.items():
        for test_name in test_names:
            input_path = get_sts_input_path(year, test_name)

            sents = read_sts_input(input_path)
            for sent in sents:
                for word in sent:
                    sts_words.add(word)

    create_glove_subset(sts_words)


def eval_sts_all(emb_func):
    '''
    Evaluates given embedding function on all STS12-STS16 files.

    Writes results in a new CSV file in LOG_PATH directory.
    '''
    create_glove_sts_subset()

    year_names = []
    test_names = []
    results = []
    for year in TEST_NAMES:
        # evaluate on STS sets from given year
        n_tests = len(TEST_NAMES[year])
        year_res = eval_sts_year(emb_func, year)
        assert len(year_res) == n_tests

        # update lists with results
        year_names.append('STS{}'.format(year))
        year_names.extend(['' for _ in range(n_tests - 1)])
        test_names.extend(TEST_NAMES[year])
        results.extend(year_res)

    # write complete log file
    file_name = 'STS-ALL-{}.csv'.format(get_cur_time_str())
    file_path = LOG_PATH.joinpath(file_name)
    with open(file_path, 'w+') as log_file:
        writer = csv.writer(log_file, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(year_names)
        writer.writerow(test_names)
        writer.writerow(['{:.3f}'.format(res) for res in results])
    print('Complete results are in file\n{}\n'.format(file_path))

