#!/bin/bash

GLOVE_DIR="../resources/embeddings/glove"
GLOVE_URL="http://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_ZIPFILE="glove.840B.300d.zip"


echo "Checking for embeddings:"

if [ -d $GLOVE_DIR ]; then
    echo "Found GloVe embeddings"
else
    echo "GloVe embeddings not found"

    mkdir -p $GLOVE_DIR || exit $?
    cd $GLOVE_DIR

    echo "Downloading from $GLOVE_URL"
    curl -LO $GLOVE_URL

    echo "Extracting into $GLOVE_DIR"
    unzip $GLOVE_ZIPFILE
    rm $GLOVE_ZIPFILE
fi
