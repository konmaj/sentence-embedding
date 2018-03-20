#!/usr/bin/env python3

from sent_emb.algorithms import glove_embeddings_mean, simpleSVD, simple_autoencoder, doc2vec, fasttext_mean
from sent_emb.statistics.statistics import all_statistics
from sent_emb.downloader import downloader
from sent_emb.evaluation import sts
from preprocessing import PreprocessingNltk, PreprocessingStanford

downloader.get_datasets()

#all_statistics(PreprocessingNltk())
#sts.eval_sts_all(doc2vec.embeddings, PreprocessingNltk(), doc2vec.train)
sts.eval_sts_all(fasttext_mean.embeddings, PreprocessingNltk())
