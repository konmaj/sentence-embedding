#!/bin/bash

STANFORD_DIR="../resources/stanford"
STANFORD_URL="http://nlp.stanford.edu/software/"
PARSER_ZIP="stanford-parser-full-2015-04-20.zip"
POSTAGGER_ZIP="stanford-postagger-full-2015-04-20.zip"
NER_ZIP="stanford-ner-2015-04-20.zip"

echo "Checking for StanfordNLP package:"

if [ -d $STANFORD_DIR ]; then
    echo "Found StanfordNLP package"
else
    echo "StanfordNLP package not found"

    mkdir -p $STANFORD_DIR || exit $?
    cd $STANFORD_DIR

    echo "Downloading from $STANFORD_URL"
    wget $STANFORD_URL$PARSER_ZIP
    wget $STANFORD_URL$POSTAGGER_ZIP
    wget $STANFORD_URL$NER_ZIP

    echo "Extracting into $STANFORD_DIR"
    for zip in $PARSER_ZIP $POSTAGGER_ZIP $NER_ZIP
    do
        unzip $zip
        rm $zip
    done
fi