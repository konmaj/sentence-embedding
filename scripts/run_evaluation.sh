#!/bin/bash

cd "$(dirname "$0")"

./get_embeddings.sh || exit $?
./get_stanfordNLP.sh || exit $?

docker build -t sentence-embedding ..


docker run \
   -v "$(pwd)/../src/sent_emb:/opt/sent_emb" \
   -v "$(pwd)/../resources/:/opt/resources" \
   -w /opt/sent_emb/evaluation \
   -e PYTHONPATH=/opt/ \
   sentence-embedding\
      evaluation.py
