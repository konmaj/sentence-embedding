#!/usr/bin/env python3

from sent_emb.algorithms import glove_embeddings_mean
from sent_emb.downloader import downloader

downloader.get_datasets()
downloader.get_embeddings()
