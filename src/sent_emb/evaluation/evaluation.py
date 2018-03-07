#!/usr/bin/env python3
import argparse

from sent_emb.algorithms import glove_embeddings_mean, simpleSVD, simple_autoencoder, doc2vec
from sent_emb.statistics.statistics import all_statistics
from sent_emb.downloader import downloader
from sent_emb.evaluation import sts
from preprocessing import PreprocessingNltk, PreprocessingStanford


# All available evaluation methods.
EVALUATIONS = ['STS']

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
parser.add_argument('-e', '--evaluation', help='select evaluation method',
                    choices=EVALUATIONS, default='STS')
parser.add_argument('-t', '--tokenizer', help='select tokenizer',
                    choices=TOKENIZERS.keys(), default='NLTK')
parser.add_argument('algorithm', type=str, help='select algorithm to run',
                    choices=ALGORITHMS.keys())
args = parser.parse_args()

params_msg = '''
Evaluation params
       method: {0}
    tokenizer: {1}
    algorithm: {2}
'''.format(args.evaluation, args.tokenizer, args.algorithm)
print(params_msg)


# Prepare evaluation

downloader.get_datasets()

tokenizer = TOKENIZERS[args.tokenizer]()
algorithm = ALGORITHMS[args.algorithm]()


# Run proper evaluation function

if args.evaluation == 'STS':
    sts.eval_sts_all(algorithm, tokenizer)
else:
    # Place here new evaluation methods if such will arise.
    assert False


#all_statistics(PreprocessingNltk())
