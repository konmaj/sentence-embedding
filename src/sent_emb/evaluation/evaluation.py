#!/usr/bin/env python3

from sent_emb.algorithms import glove_embeddings_mean, simpleSVD
from sent_emb.downloader import downloader
from sent_emb.evaluation import sts

downloader.get_datasets()
downloader.get_embeddings()

sts.eval_sts_all(simpleSVD.embeddings)
