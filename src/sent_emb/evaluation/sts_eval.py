import csv
import datetime
import subprocess
import re
import numpy as np

from sent_emb.algorithms.path_utility import RESOURCES_DIR, DATASETS_DIR
from sent_emb.downloader.downloader import mkdir_if_not_exist
from sent_emb.evaluation.model import (BaseAlgorithm, flatten_sent_pairs)
from sent_emb.evaluation.sts_read import (TEST_NAMES, STS, get_sts_input_path, get_sts_gs_path,
                                          get_sts_output_path, read_sts_input, read_train_set)

LOG_PATH = RESOURCES_DIR.joinpath('log')


# Script from STS16 seems to be backward compatible with file formats from former years.
GRADING_SCRIPT_PATH = DATASETS_DIR.joinpath('STS16', 'test-data', 'correlation-noconfidence.pl')


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


def get_cur_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def generate_similarity_file(algorithm, input_path, output_path, tokenizer):
    """
    Runs given embedding algorithm.transform() method on a single STS task (without
    computing score).

    Writes output in format described in section Output Files of file
    resources/datasets/STS16/test-data/README.txt
    """
    # read test data
    sent_pairs = read_sts_input(input_path, tokenizer)
    sents = flatten_sent_pairs(sent_pairs)

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


def eval_sts_year(year, algorithm, tokenizer, year_file=False, smoke_test=False, training=True):
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

    if training:
        print('Reading training set for', sts_name)
        train_sents = read_train_set(year, tokenizer)

        train_sents = flatten_sent_pairs(train_sents)

        if smoke_test:
            train_sents = train_sents[:10]
        print('numbers of sentences:', len(train_sents))
        print('Training started...')
        algorithm.fit(train_sents)
        print('... training completed.')
    else:
        print('No training')

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


def eval_sts_all(algorithm, tokenizer, years_choose=TEST_NAMES.keys(), training=True):
    """
    Evaluates given embedding algorithm on all STS12-STS16 files.

    Writes results in a new CSV file in LOG_PATH directory.

    algorithm: instance of BaseAlgorithm class
               (see docstring of sent_emb.evaluation.model.BaseAlgorithm for more info)

    years_choose: specifies years to choose datasets from

    training: if False, then skips phase of training
        (in particular skips loading resources for training)
    """
    dataset = STS(tokenizer)
    algorithm.get_resources(dataset)

    year_names = []
    test_names = []
    results = []

    sparse_names = []
    sparse_results = []

    years = {}
    for year in years_choose:
        years[year] = TEST_NAMES[year]

    for year in sorted(years):
        # evaluate on STS sets from given year
        n_tests = len(years[year])
        year_res = eval_sts_year(year, algorithm, tokenizer, training=training)
        assert len(year_res) == n_tests
        year_avg = sum(year_res) / n_tests

        # update lists with results
        year_names.extend(['STS{}'.format(year)] + ['' for _ in range(n_tests)])
        test_names.extend(years[year] + ['avg'])
        sparse_names.append('STS {} avg'.format(year))
        results.extend(year_res + [year_avg])
        sparse_results.append(year_avg)

    # write complete log file
    file_name = 'STS-ALL-{}.csv'.format(get_cur_time_str())
    file_path = LOG_PATH.joinpath(file_name)
    with open(str(file_path), 'w+') as log_file:
        writer = csv.writer(log_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        writer.writerow(year_names)
        writer.writerow(test_names)
        writer.writerow(['{:.1f}'.format(res) for res in results])

    file_name_sparse = 'STS-SPARSE-{}.csv'.format(get_cur_time_str())
    file_path_sparse = LOG_PATH.joinpath(file_name_sparse)
    with open (str(file_path_sparse), "w+") as log_file:
        writer = csv.writer(log_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        writer.writerow(sparse_names)
        writer.writerow(['{:.1f}'.format(res) for res in sparse_results])

    print('Complete results are in files\n{}\n{}\n'.format(file_path, file_path_sparse))
