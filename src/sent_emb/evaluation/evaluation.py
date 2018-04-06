#!/usr/bin/env python3
import argparse
import json
import sys

from sent_emb.algorithms import (glove_embeddings_mean, simpleSVD,
                                 simple_autoencoder, doc2vec,
                                 fasttext_mean, seq2seq)
from sent_emb.statistics.statistics import all_statistics
from sent_emb.downloader import downloader
from sent_emb.evaluation import sts_eval
from sent_emb.evaluation.preprocessing import PreprocessingNltk, PreprocessingStanford


# All available evaluation methods.
RUN_MODES = ['STS', 'stats', 'test', 'train_s2s']

# All available algorithms with their constructors.
ALGORITHMS = {
    'Doc2Vec': doc2vec.Doc2Vec,
    'GloveMean': glove_embeddings_mean.GloveMean,
    'SVD': simpleSVD.SimpleSVD,
    'Autoencoder': simple_autoencoder.SimpleAutoencoder,
    'Seq2Seq': seq2seq.Seq2Seq,
    'FastTextMean': fasttext_mean.FastTextMean,
    'FastTextSVD': fasttext_mean.FastTextSVD,
    'FastTextMeanWithoutUnknown': fasttext_mean.FastTextMeanWithoutUnknown,
}

# Algorithms excluded from the smoke test
EXCLUDED_FROM_TEST = ['Seq2Seq']

# All available tokenizers with their constructors.
TOKENIZERS = {
    'NLTK': PreprocessingNltk,
    'Stanford': PreprocessingStanford
}


# Parse parameters

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run-mode', help='select run mode of script',
                    choices=RUN_MODES, default='STS')
parser.add_argument('-t', '--tokenizer', help='select tokenizer',
                    choices=TOKENIZERS.keys(), default='NLTK')
parser.add_argument('algorithm', nargs='?', type=str, help='select algorithm to run',
                    choices=ALGORITHMS.keys())
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
    alg-kwargs: {3}
'''.format(args.run_mode, args.tokenizer, args.algorithm, alg_kwargs)
    print(params_msg)

    if args.algorithm is None:
        print('\nERROR: In \'STS\' mode you have to specify algorithm to assess.')
        print('Example: ./run_evaluation.sh Doc2Vec')
        sys.exit(1)

    tokenizer = TOKENIZERS[args.tokenizer]()
    algorithm = ALGORITHMS[args.algorithm](**alg_kwargs)
    sts_eval.eval_sts_all(algorithm, tokenizer)

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
        sts_eval.eval_sts_year(12, algorithm, token_ctor(), smoke_test=True)

    tokenizer = TOKENIZERS['NLTK']()  # take any tokenizer
    for algo_name, algo_ctor in ALGORITHMS.items():
        print('\nTesting algorithm {}...\n'.format(algo_name))
        if algo_name in EXCLUDED_FROM_TEST:
            print('...algorithm {} is excluded from the test - skip.'.format(algo_name))
        else:
            sts_eval.eval_sts_year(12, algo_ctor(), tokenizer, smoke_test=True)

elif args.run_mode == 'train_s2s':
    alg_kwargs = json.loads(args.alg_kwargs)
    params_msg = '''
 Script params
    run-mode: {0}
    alg-kwargs: {1}
'''.format(args.run_mode, alg_kwargs)
    print(params_msg)

    seq2seq.improve_model(ALGORITHMS['Seq2Seq'](**alg_kwargs),
                          TOKENIZERS[args.tokenizer]())

else:
    assert False
