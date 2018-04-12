#!/bin/bash

# Adjust variables below according to properties of your machine (see doc/README_docker.md)
PHYSICAL_MEMORY_LIMIT=4g
VIRTUAL_MEMORY_LIMIT=11g # physical memory + disk swap

cd "$(dirname "$0")"

./get_stanfordNLP.sh || exit $?

docker build -t sentence-embedding ..

RESOURCES_MOUNT_DIR=/opt/resources

docker run \
   -v "$(pwd)/../src/sent_emb:/opt/sent_emb" \
   -v "$(pwd)/../resources/:${RESOURCES_MOUNT_DIR}" \
   -w /opt/sent_emb/evaluation \
   -e PYTHONPATH=/opt/ \
   -e RESOURCES_DIR=${RESOURCES_MOUNT_DIR} \
   --memory ${PHYSICAL_MEMORY_LIMIT} \
   --memory-swap ${VIRTUAL_MEMORY_LIMIT} \
   sentence-embedding\
      evaluation.py "$@"
