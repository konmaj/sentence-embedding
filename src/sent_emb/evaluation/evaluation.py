#!/usr/bin/env python3

import nltk

from sent_emb.algorithms import glove_embeddings_mean
from sent_emb.downloader import downloader
from sent_emb.evaluation import sts

downloader.get_datasets()
downloader.get_embeddings()
downloader.get_resources()

nltk.download('punkt')

sts.eval_sts_all(glove_embeddings_mean.embeddings)
