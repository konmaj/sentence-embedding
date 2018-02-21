import csv
import datetime
import subprocess
import re

from shutil import copyfile

import numpy as np
from pathlib import Path

from sent_emb.algorithms.glove_utility import create_glove_subset, get_glove_file, GLOVE_FILE
from sent_emb.downloader.downloader import mkdir_if_not_exist

STS12_TRAIN_NAMES = ['MSRpar', 'MSRvid', 'SMTeuroparl']

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
GRADING_SCRIPT_PATH = DATASETS_PATH.joinpath('STS16', 'test-data', 'correlation-noconfidence.pl')


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


def get_sts_input_path(year, test_name, use_train_set=False):
    assert year in TEST_NAMES
    assert not (use_train_set and year != 12)

    sts_path = get_sts_path(year)
    dir_name = 'test-data' if not use_train_set else 'train-data'
    input_name = 'STS.input.{}.txt'.format(test_name)

    return sts_path.joinpath(dir_name, input_name)


def get_sts_gs_path(year, test_name):
    assert year in TEST_NAMES

    sts_path = get_sts_path(year)
    gs_name = 'STS.gs.{}.txt'.format(test_name)

    return sts_path.joinpath('test-data', gs_name)


def get_sts_output_path(year, test_name):
    assert year in TEST_NAMES

    sts_path = get_sts_path(year)
    output_name = 'STS.output.{}.txt'.format(test_name)

    return sts_path.joinpath('out', output_name)


def get_cur_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def tokens(tokenizer, sents):
    guard = "verylongwordwhichisntawordanddoesntappearinlanguage"
    con = ''
    for sent in sents:
        con = con + sent + '\n' + guard + '\n'
    tokenized = tokenizer.tokenize(con)
    res = [[]]
    for word in tokenized:
        if word == guard:
            res.append([])
        else:
            res[-1].append(word)
    return res[:-1]

def read_sts_input(file_path, tokenizer):
    '''
    Reads STS input file at given 'file_path'.

    returns: numpy array of sentences
        sentence: python list of words
        word: string
        TODO: unify return type - only numpy or only python lists.
    '''
    sents = []
    with open(file_path, 'r') as test_file:
        test_reader = csv.reader(test_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in test_reader:
            assert len(row) == 2 \
                or len(row) == 4 # STS16 contains also source of each sentence
            sents.extend(row[:2])
    return np.array(tokens(tokenizer, sents))


def read_train_set(year, tokenizer):
    '''
    Reads training set available for STS in given 'year'.

    For each year training set consists of:
    1) STS12 train-data
    2) Test data from STS from former years

    returns: sequence of sentences from all training sets available for STS in given 'year'
        type: consistent with concatenation of results of function 'read_sts_input'
    '''
    # STS12 train-data...
    train_inputs = []
    for test_name in STS12_TRAIN_NAMES:
        input_path = get_sts_input_path(12, test_name, use_train_set=True)
        train_inputs.append(read_sts_input(input_path, tokenizer))

    # test sets from STS before given 'year'
    for test_year, test_names_year in sorted(TEST_NAMES.items()):
        if test_year >= year:
            break
        for test_name in test_names_year:
            input_path = get_sts_input_path(test_year, test_name, use_train_set=False)
            train_inputs.append(read_sts_input(input_path, tokenizer))

    return np.concatenate(tuple(train_inputs))


def generate_similarity_file(emb_func, input_path, output_path, tokenizer):
    '''
    Runs given embedding function ('emb_func') on a single STS task (without
    computing score).

    Writes output in format described in section Output Files of file
    resources/datasets/STS16/data/README.txt
    '''
    # read test data
    sents = read_sts_input(input_path, tokenizer)

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
    res = re.search(r'^Pearson: (-?\d\.\d{5})$', output)
    assert res is not None
    return float(res.groups()[0]) # throws exception in case of wrong conversion


def eval_sts_year(year, emb_func, tokenizer, train_func=None, year_file=False):
    '''
    Evaluates given embedding function on STS inputs from given year.

    1) Trains given algorithm by applying 'train_func' to train dataset
       (only if 'train_func' is not None).
    2) Evaluates algorithm by applying 'emb_func' to various test datasets

    Signatures:
        emb_func(sents)
            sents: sequence of sentences as in 'read_sts_input' function result
            returns: numpy 2-D array of sentence embeddings
        train_func(sents)
            sents: sequence of sentences as in 'read_sts_input' function result
            returns: None

    If year_file=True, generates file with results in the LOG_PATH directory.

    returns: list of "Pearson's r * 100" of each input
             (ordered as in TEST_NAMES[year]).
    '''
    assert year in TEST_NAMES
    sts_name = 'STS{}'.format(year)

    print('Reading training set for', sts_name)
    train_sents = None
    if train_func is None:
        print('... no function for training provided - training skipped.')
    else:
        train_sents = read_train_set(year, tokenizer)
        print('numbers of sentences:', train_sents.shape[0])
        print('Training started...')
        train_func(train_sents)
        print('... training completed.')

    print('Evaluating on datasets from', sts_name)
    results = []

    mkdir_if_not_exist(LOG_PATH)
    cur_time = get_cur_time_str()
    log_file_name = 'STS{}-{}.txt'.format(year, cur_time)
    log_file_path = LOG_PATH.joinpath(log_file_name)

    for test_name in TEST_NAMES[year]:

        print('Evaluating on test {} from {}'.format(test_name, sts_name))

        # generate out
        in_path = get_sts_input_path(year, test_name)
        out_path = get_sts_output_path(year, test_name)
        gs_path = get_sts_gs_path(year, test_name)

        generate_similarity_file(emb_func, in_path, out_path, tokenizer)

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


def create_glove_sts_subset(tokenizer):
    '''
    1) Computes set of words which appeared in STS input files.
    2) Creates reduced GloVe file, which contains only words that appeared
       in STS.
    '''
    if glove_file(tokenizer.name()).exists():
        print('Cropped GloVe file already exists')
    else:
        sts_words = set()
        for year, test_names in sorted(TEST_NAMES.items()):
            for test_name in test_names:
                input_path = get_sts_input_path(year, test_name)

                sents = read_sts_input(input_path, tokenizer)
                for sent in sents:
                    for word in sent:
                        sts_words.add(word)

        create_glove_subset(sts_words, tokenizer.name())

    copyfile(glove_file(tokenizer.name()), GLOVE_FILE)


def eval_sts_all(emb_func, tokenizer, train_func=None):
    '''
    Evaluates given embedding algorithm on all STS12-STS16 files.

    Writes results in a new CSV file in LOG_PATH directory.

    emb_func, train_func: see docstring of 'eval_sts_year' function.
    '''
    create_glove_sts_subset(tokenizer)

    year_names = []
    test_names = []
    results = []
    for year in sorted(TEST_NAMES):
        # evaluate on STS sets from given year
        n_tests = len(TEST_NAMES[year])
        year_res = eval_sts_year(year, emb_func, tokenizer, train_func)
        assert len(year_res) == n_tests
        year_avg = sum(year_res) / n_tests

        # update lists with results
        year_names.extend(['STS{}'.format(year)] + ['' for _ in range(n_tests)])
        test_names.extend(TEST_NAMES[year] + ['avg'])
        results.extend(year_res + [year_avg])

    # write complete log file
    file_name = 'STS-ALL-{}.csv'.format(get_cur_time_str())
    file_path = LOG_PATH.joinpath(file_name)
    with open(file_path, 'w+') as log_file:
        writer = csv.writer(log_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        writer.writerow(year_names)
        writer.writerow(test_names)
        writer.writerow(['{:.3f}'.format(res) for res in results])
    print('Complete results are in file\n{}\n'.format(file_path))

