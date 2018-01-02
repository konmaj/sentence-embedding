import csv
import datetime
import subprocess

import numpy as np
from pathlib import Path

from sent_emb.downloader.downloader import mkdir_if_not_exist


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
            assert len(row) == 2
            sents.extend(row)
    sents = np.array(sents)

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


def eval_sts15(emb_func):
    test_names = ['answers-forums', 'answers-students', 'belief', 'headlines', 'images']

    sts_dir = Path('/', 'opt', 'resources', 'datasets', 'STS15')
    data_dir = sts_dir.joinpath('data')
    out_dir = sts_dir.joinpath('out')
    in_prefix = 'STS.input'
    out_prefix = 'STS.output'
    gs_prefix = 'STS.gs'

    log_msg = ''

    for test_name in test_names:
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
        script = data_dir.joinpath('correlation-noconfidence.pl')

        score = subprocess.check_output(
            ['perl', script, gs_path, out_path],
            universal_newlines=True,
        )

        print(score)
        log_msg += 'test name: {}\n{}\n'.format(test_name, score)

    log_dir = Path('/', 'opt', 'resources', 'log')
    mkdir_if_not_exist(log_dir)

    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    log_name = 'STS15-{}.txt'.format(cur_time)
    with open(log_dir.joinpath(log_name), 'w+') as log_file:
        log_file.write(log_msg)
