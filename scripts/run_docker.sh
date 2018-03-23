#!/bin/bash

cd "$(dirname "$0")"

./get_embeddings.sh || exit $?
./get_stanfordNLP.sh || exit $?

docker build -t sentence-embedding ..

RESOURCES_MOUNT_DIR=/opt/resources

docker run \
   -v "$(pwd)/../src/sent_emb:/opt/sent_emb" \
   -v "$(pwd)/../resources/:${RESOURCES_MOUNT_DIR}" \
   -w /opt/sent_emb/evaluation \
   -e PYTHONPATH=/opt/ \
   -e RESOURCES_DIR=${RESOURCES_MOUNT_DIR} \
   sentence-embedding\
      evaluation.py "$@"
