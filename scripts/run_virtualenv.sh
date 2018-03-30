#!/bin/bash

set -e # stop on any error

cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)
PROJECT_DIR=${SCRIPT_DIR}/..

PIP_ENV=${PROJECT_DIR}/py35
REQ_DIR=${PROJECT_DIR}/requirements
SCRIPT_TO_RUN=${PROJECT_DIR}/src/sent_emb/evaluation/evaluation.py

# Prepare virtualenv

if [ ! -d "$PIP_ENV" ]; then
    virtualenv -p python3.5 "${PIP_ENV}"
fi

source ${PIP_ENV}/bin/activate

pip3 install -r "${REQ_DIR}/1_math.txt"
pip3 install -r "${REQ_DIR}/2_nlp.txt"
pip3 install -r "${REQ_DIR}/3_ml.txt"
pip3 install -r "${REQ_DIR}/4_other.txt"
pip3 install tensorflow-gpu==1.6.0

python3.5 -c "import nltk; nltk.download('punkt')"

# Run script with environment variables
PYTHONPATH=${PROJECT_DIR}/src \
RESOURCES_DIR=${PROJECT_DIR}/resources \
python3.5 ${SCRIPT_TO_RUN} "$@"

deactivate
