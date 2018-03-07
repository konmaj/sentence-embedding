#!/usr/bin/env python3
import argparse
import json
import sys

from sent_emb.algorithms import glove_embeddings_mean, simpleSVD, simple_autoencoder, doc2vec
from sent_emb.statistics.statistics import all_statistics
from sent_emb.downloader import downloader
from sent_emb.evaluation import sts
from preprocessing import PreprocessingNltk, PreprocessingStanford


# All available evaluation methods.
RUN_MODES = ['STS', 'test']

# All available algorithms with their constructors.
ALGORITHMS = {
    'Doc2Vec': doc2vec.Doc2Vec
}

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


alg_kwargs = json.loads(args.alg_kwargs)

if args.run_mode == 'test':
    args.tokenizer = 'ignored'
    args.algorithm = 'ignored'
    args.alg_kwargs = 'ignored'
else:
    if args.algorithm is None:
        print('\nERROR: You have to specify algorithm in every run mode except the \'test\' mode.')
        print('Example: ./run_evaluation.sh Doc2Vec')
        sys.exit(1)

params_msg = '''
Evaluation params
      run-mode: {0}
     tokenizer: {1}
     algorithm: {2}
    alg-kwargs: {3}
'''.format(args.run_mode, args.tokenizer, args.algorithm, alg_kwargs)
print(params_msg)


# Run proper evaluation function

downloader.get_datasets()

if args.run_mode == 'STS':
    tokenizer = TOKENIZERS[args.tokenizer]()
    algorithm = ALGORITHMS[args.algorithm](**alg_kwargs)
    sts.eval_sts_all(algorithm, tokenizer)

elif args.run_mode == 'test':
    algorithm = ALGORITHMS['Doc2Vec']() # take any algorithm
    for token_name, token_ctor in TOKENIZERS.items():
        print('\nTesting tokenizer {}...\n'.format(token_name))
        sts.eval_sts_year(12, algorithm, token_ctor(), smoke_test=True)

    tokenizer = TOKENIZERS['NLTK']() # take any tokenizer
    for algo_name, algo_ctor in ALGORITHMS.items():
        print('\nTesting algorithm {}...\n'.format(algo_name))
        sts.eval_sts_year(12, algo_ctor(), tokenizer, smoke_test=True)

else:
    assert False


#all_statistics(PreprocessingNltk())
