#!/usr/bin/env python3
import argparse
import json
import sys

import sent_emb.algorithms.seq2seq.autoencoder
from sent_emb.algorithms import (glove_embeddings_mean, simpleSVD,
                                 simple_autoencoder, doc2vec,
                                 fasttext_mean)
from sent_emb.algorithms.seq2seq.utility import improve_model
from sent_emb.algorithms.seq2seq import autoencoder, autoencoder_with_cosine
from sent_emb.statistics.statistics import all_statistics
from sent_emb.downloader import downloader
from sent_emb.evaluation import sts_eval, sts_read
from sent_emb.evaluation.preprocessing import PreprocessingNltk, PreprocessingStanford


# All available evaluation methods.
RUN_MODES = ['STS', 'stats', 'test', 'train_s2s']

# All available algorithms with their constructors.
ALGORITHMS = {
    'Doc2Vec': doc2vec.Doc2Vec,
    'GloveMean': glove_embeddings_mean.GloveMean,
    'SVD': simpleSVD.SimpleSVD,
    'Autoencoder': simple_autoencoder.SimpleAutoencoder,
    'S2SAutoencoder': autoencoder.Autoencoder,
    'S2SAutoencoderWithCosine': autoencoder_with_cosine.AutoencoderWithCosine,
    'FastTextMean': fasttext_mean.FastTextMean,
    'FastTextSVD': fasttext_mean.FastTextSVD,
    'FastTextMeanWithoutUnknown': fasttext_mean.FastTextMeanWithoutUnknown,
}

# Algorithms excluded from the smoke test
EXCLUDED_FROM_TEST = ['S2SAutoencoder', 'S2SAutoencoderWithCosine', 'FastTextMean', 'FastTextSVD',
                      'FastTextMeanWithoutUnknown']

# All available tokenizers with their constructors.
TOKENIZERS = {
    'NLTK': PreprocessingNltk,
    'Stanford': PreprocessingStanford
}

YEARS = ['*', '12', '13', '14', '15', '16']


# Parse parameters

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run-mode', help='select run mode of script',
                    choices=RUN_MODES, default='STS')
parser.add_argument('-t', '--tokenizer', help='select tokenizer',
                    choices=TOKENIZERS.keys(), default='NLTK')
parser.add_argument('algorithm', nargs='?', type=str, help='select algorithm to run',
                    choices=ALGORITHMS.keys())
parser.add_argument('-y', '--year', help='select STS year',
                    choices=YEARS, default='*')
parser.add_argument('--alg-kwargs', help='specify JSON with kwargs to init method of algorithm',
                    default='{}')
args = parser.parse_args()


# Run proper mode

downloader.get_datasets()

if args.run_mode == 'STS':
    alg_kwargs = json.loads(args.alg_kwargs)

    params_msg = '''
Script params
    run-mode: {0}
    tokenizer: {1}
    algorithm: {2}
    year: {3}
    alg-kwargs: {4}
'''.format(args.run_mode, args.tokenizer, args.algorithm, args.year, alg_kwargs)
    print(params_msg)

    if args.algorithm is None:
        print('\nERROR: In \'STS\' mode you have to specify algorithm to assess.')
        print('Example: ./run_docker.sh Doc2Vec')
        sys.exit(1)

    tokenizer = TOKENIZERS[args.tokenizer]()
    algorithm = ALGORITHMS[args.algorithm](**alg_kwargs)
    
    if args.year == '*':
        sts_eval.eval_sts_all(algorithm, tokenizer)
    else:
        sts_eval.eval_sts_all(algorithm, tokenizer, [int(args.year)])

elif args.run_mode == 'stats':
    params_msg = '''
Script params
    run-mode: {0}
    tokenizer: {1}
'''.format(args.run_mode, args.tokenizer)
    print(params_msg)

    tokenizer = TOKENIZERS[args.tokenizer]()
    all_statistics(tokenizer)

elif args.run_mode == 'test':
    params_msg = '''
Script params
    run-mode: {0}
'''.format(args.run_mode)
    print(params_msg)

    algorithm = ALGORITHMS['Doc2Vec']()  # take any algorithm
    for token_name, token_ctor in TOKENIZERS.items():
        print('\nTesting tokenizer {}...\n'.format(token_name))
        tokenizer = token_ctor()
        dataset = sts_read.STS(tokenizer)
        algorithm.get_resources(tokenizer)
        sts_eval.eval_sts_year(12, algorithm, tokenizer, smoke_test=True)

    tokenizer = TOKENIZERS['NLTK']()  # take any tokenizer
    dataset = sts_read.STS(tokenizer)
    for algo_name, algo_ctor in ALGORITHMS.items():
        print('\nTesting algorithm {}...\n'.format(algo_name))
        if algo_name in EXCLUDED_FROM_TEST:
            print('...algorithm {} is excluded from the test - skip.'.format(algo_name))
        else:
            algorithm = algo_ctor()
            algorithm.get_resources(dataset)
            sts_eval.eval_sts_year(12, algorithm, tokenizer, smoke_test=True)

elif args.run_mode == 'train_s2s':
    alg_kwargs = json.loads(args.alg_kwargs)
    params_msg = '''
 Script params
    run-mode: {0}
    algorithm: {1}
    alg-kwargs: {2}
'''.format(args.run_mode, args.algorithm, alg_kwargs)
    print(params_msg)

    if args.algorithm is None:
        print('\nERROR: In \'train_s2s\' mode you have to specify algorithm to train.')
        print('Example: ./run_docker.sh -r train_s2s S2SAutoencoder')
        sys.exit(1)

    improve_model(ALGORITHMS[args.algorithm](**alg_kwargs),
                  TOKENIZERS[args.tokenizer]())

else:
    assert False
