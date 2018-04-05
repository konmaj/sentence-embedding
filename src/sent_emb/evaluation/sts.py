import csv
import datetime
import subprocess
import re
import numpy as np
from collections import namedtuple

from sent_emb.algorithms.glove_utility import create_glove_subset, get_glove_file, GLOVE_FILE
from sent_emb.algorithms.path_utility import RESOURCES_DIR, DATASETS_DIR
from sent_emb.downloader.downloader import mkdir_if_not_exist
from sent_emb.evaluation.model import BaseAlgorithm, DataSet
from sent_emb.algorithms.fasttext_utility import fasttext_preprocessing

STS12_TRAIN_NAMES = ['MSRpar', 'MSRvid', 'SMTeuroparl']

TEST_NAMES = {
    12: ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews'],
    13: ['headlines', 'OnWN', 'FNWN'],
    14: ['deft-forum', 'deft-news', 'headlines', 'images', 'OnWN', 'tweet-news'],
    15: ['answers-forums', 'answers-students', 'belief', 'headlines', 'images'],
    16: ['answer-answer', 'headlines', 'plagiarism', 'postediting', 'question-question'],
}

LOG_PATH = RESOURCES_DIR.joinpath('log')


# Script from STS16 seems to be backward compatible with file formats from former years.
GRADING_SCRIPT_PATH = DATASETS_DIR.joinpath('STS16', 'test-data', 'correlation-noconfidence.pl')


SentPair = namedtuple('SentPair', ['sent1', 'sent2'])

SentPairWithGs = namedtuple('SentPairWithGs', SentPair._fields + ('gs',))


def zip_sent_pairs_with_gs(sent_pairs, gold_standards):
    assert len(sent_pairs) == len(gold_standards)
    return [SentPairWithGs(*pair, gs=gs) for pair, gs in zip(sent_pairs, gold_standards)]


def flatten_sent_pairs(sent_pairs):
    return [sent for sent_pair in sent_pairs for sent in [sent_pair.sent1, sent_pair.sent2]]


class STS(DataSet):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def word_set(self):
        if not hasattr(self, '_word_set_value'):
            input_paths = [get_sts_input_path(12, train_name, use_train_set=True)
                           for train_name in STS12_TRAIN_NAMES]
            input_paths += [get_sts_input_path(year, test_name, use_train_set=False)
                            for year, test_names in sorted(TEST_NAMES.items())
                            for test_name in test_names]

            sts_words = set()
            for input_path in input_paths:
                sent_pairs = read_sts_input(input_path, self.tokenizer)
                sents = flatten_sent_pairs(sent_pairs)
                for sent in sents:
                    for word in sent:
                        sts_words.add(word)
            self._word_set_value = sts_words

        return self._word_set_value

    def tokenizer_name(self):
        return self.tokenizer.name()


def vector_len(vec):
    return np.sum(np.square(vec)) ** 0.5


def cos(vec0, vec1):
    dot_product = np.sum(vec0 * vec1)
    return dot_product / (vector_len(vec0) * vector_len(vec1))


def compute_similarity(emb_pairs):
    result = []

    for pair in emb_pairs:
        sim = cos(pair[0], pair[1])
        result.append((sim + 1) * 5 / 2)  # scale interval from [-1; 1] to [0; 5]

    return np.array(result)


def get_sts_path(year):
    assert year in TEST_NAMES
    return DATASETS_DIR.joinpath('STS{}'.format(year))


def get_sts_input_path(year, test_name, use_train_set=False):
    assert year in TEST_NAMES
    assert not (use_train_set and year != 12)

    sts_path = get_sts_path(year)
    dir_name = 'test-data' if not use_train_set else 'train-data'
    input_name = 'STS.input.{}.txt'.format(test_name)

    return sts_path.joinpath(dir_name, input_name)


def get_sts_gs_path(year, test_name, use_train_set=False):
    assert year in TEST_NAMES
    assert not (use_train_set and year != 12)

    sts_path = get_sts_path(year)
    dir_name = 'test-data' if not use_train_set else 'train-data'
    gs_name = 'STS.gs.{}.txt'.format(test_name)

    return sts_path.joinpath(dir_name, gs_name)


def get_sts_output_path(year, test_name):
    assert year in TEST_NAMES

    sts_path = get_sts_path(year)
    output_name = 'STS.output.{}.txt'.format(test_name)

    return sts_path.joinpath('out', output_name)


def get_cur_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def tokens(tokenizer, sents):
    """
    Tokenizes each sentence in a list.

    :param tokenizer: tokenizer to use
    :param sents: list of sentences (strings) to tokenize
    :return: list of tokenized sentences - each sentence is represented as a list of words
    """
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
    """
    Reads STS input file at given `file_path`.

    returns: list of SentPairs
    """
    sents = []
    with open(str(file_path), 'r') as test_file:
        test_reader = csv.reader(test_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in test_reader:
            assert len(row) == 2 \
                or len(row) == 4  # STS16 contains also source of each sentence
            sents.extend(row[:2])

    sents = tokens(tokenizer, sents)

    return [SentPair(sent1, sent2) for sent1, sent2 in zip(sents[::2], sents[1::2])]


def read_sts_gs(file_path):
    """
    Reads STS gold standard file at given 'file_path'.

    :param file_path: Path to gold standard file
    :return: list of gold standard scores (floats) for pairs of sentences
        Missing scores are represented as `None` in the resulting list.
    """
    gs_score = []
    with open(str(file_path), 'r') as gs_file:
        for line in gs_file.readlines():
            assert line[-1] == '\n'
            gs_score.append(float(line[:-1]) if len(line) > 1 else None)
    return gs_score


def read_sts_input_with_gs(year, test_name, tokenizer, use_train_set=False):
    input_path = get_sts_input_path(year, test_name, use_train_set=use_train_set)
    sents_pairs = read_sts_input(input_path, tokenizer)

    gs_path = get_sts_gs_path(year, test_name, use_train_set=use_train_set)
    gold_standards = read_sts_gs(gs_path)

    return zip_sent_pairs_with_gs(sents_pairs, gold_standards)


def read_train_set(year, tokenizer):
    """
    Reads training set available for STS in given 'year'.

    For each year training set consists of:
    1) STS12 train-data
    2) Test data from STS from former years

    :param year: two last digits of year of STS task (e.g. 12)
    :param tokenizer: tokenizer to use while reading sentences
    :return: list of tuples with training data available for STS in given `year`
        tuple: (sentence1, sentence2, gold_standard)
        sentence: list of words (strings)
    """
    # STS12 train-data...
    train_data = []
    for test_name in STS12_TRAIN_NAMES:
        train_data.extend(read_sts_input_with_gs(12, test_name, tokenizer, use_train_set=True))

    # test sets from STS before given 'year'
    for test_year, test_names_year in sorted(TEST_NAMES.items()):
        if test_year >= year:
            break
        for test_name in test_names_year:
            train_data.extend(read_sts_input_with_gs(test_year, test_name, tokenizer,
                                                     use_train_set=False))

    return train_data


def generate_similarity_file(algorithm, input_path, output_path, tokenizer):
    """
    Runs given embedding algorithm.transform() method on a single STS task (without
    computing score).

    Writes output in format described in section Output Files of file
    resources/datasets/STS16/data/README.txt
    """
    # read test data
    sent_pairs = read_sts_input(input_path, tokenizer)
    sents = flatten_sent_pairs(sent_pairs)
    sents = np.array(sents)  # TODO: remove conversion to numpy

    # compute embeddings
    embs = algorithm.transform(sents)
    assert len(embs) == len(sents)

    # generate similarities between pairs of sentences
    embs.shape = (embs.shape[0] // 2, 2, embs.shape[1])
    similarities = compute_similarity(embs)

    # write file with similarities
    with open(str(output_path), 'w+') as out_file:
        for sim in similarities:
            out_file.write('{}\n'.format(sim))


def get_grad_script_res(output):
    res = re.search(r'^Pearson: (-?\d\.\d{5})$', output)
    assert res is not None
    return float(res.groups()[0])  # throws exception in case of wrong conversion


def eval_sts_year(year, algorithm, tokenizer, year_file=False, smoke_test=False):
    """
    Evaluates given embedding algorithm on STS inputs from given year.

    1) Trains given algorithm by calling algorithm.fit() method on train dataset
    2) Evaluates algorithm by calling algorithm.transform() on various test datasets

    If year_file=True, generates file with results in the LOG_PATH directory.

    If smoke_test=True, shrinks training set to the first 10 sentences.

    algorithm: instance of BaseAlgorithm class
               (see docstring of sent_emb.evaluation.model.BaseAlgorithm for more info)

    returns: list of "Pearson's r * 100" of each input
             (ordered as in TEST_NAMES[year]).
    """
    assert year in TEST_NAMES
    sts_name = 'STS{}'.format(year)

    assert isinstance(algorithm, BaseAlgorithm)

    if smoke_test:
        # Get resources, since it's smoke test and it hadn't been downloaded
        dataset = STS(tokenizer)
        algorithm.get_resources(dataset)

    print('Reading training set for', sts_name)
    train_sents = read_train_set(year, tokenizer)

    train_sents = flatten_sent_pairs(train_sents)
    train_sents = np.array(train_sents)  # TODO: remove numpy

    if smoke_test:
        train_sents = train_sents[:10]
    print('numbers of sentences:', train_sents.shape[0])
    print('Training started...')
    algorithm.fit(train_sents)
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

        generate_similarity_file(algorithm, in_path, out_path, tokenizer)

        # compare out with gold standard
        script = GRADING_SCRIPT_PATH
        output = subprocess.check_output(
            ['perl', str(script), str(gs_path), str(out_path)],
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


def eval_sts_all(algorithm, tokenizer):
    """
    Evaluates given embedding algorithm on all STS12-STS16 files.

    Writes results in a new CSV file in LOG_PATH directory.

    algorithm: instance of BaseAlgorithm class
               (see docstring of sent_emb.evaluation.model.BaseAlgorithm for more info)
    """
    dataset = STS(tokenizer)
    algorithm.get_resources(dataset)

    year_names = []
    test_names = []
    results = []
    for year in sorted(TEST_NAMES):
        # evaluate on STS sets from given year
        n_tests = len(TEST_NAMES[year])
        year_res = eval_sts_year(year, algorithm, tokenizer)
        assert len(year_res) == n_tests
        year_avg = sum(year_res) / n_tests

        # update lists with results
        year_names.extend(['STS{}'.format(year)] + ['' for _ in range(n_tests)])
        test_names.extend(TEST_NAMES[year] + ['avg'])
        results.extend(year_res + [year_avg])

    # write complete log file
    file_name = 'STS-ALL-{}.csv'.format(get_cur_time_str())
    file_path = LOG_PATH.joinpath(file_name)
    with open(str(file_path), 'w+') as log_file:
        writer = csv.writer(log_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        writer.writerow(year_names)
        writer.writerow(test_names)
        writer.writerow(['{:.3f}'.format(res) for res in results])
    print('Complete results are in file\n{}\n'.format(file_path))
