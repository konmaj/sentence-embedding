#!/bin/bash

set -e # stop on any error

cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)
PROJECT_DIR=${SCRIPT_DIR}/..

PIP_ENV=${PROJECT_DIR}/py35
REQ_FILE=${PROJECT_DIR}/requirements.txt
SCRIPT_TO_RUN=${PROJECT_DIR}/src/sent_emb/evaluation/evaluation.py

if [ ! -d "$PIP_ENV" ]; then
    virtualenv -p python3.5 "${PIP_ENV}"
fi

source ${PIP_ENV}/bin/activate
pip install -r "${REQ_FILE}"
python3.5 -c "import nltk; nltk.download('punkt')"

# run script with environment variables
PYTHONPATH=${PROJECT_DIR}/src \
RESOURCES_DIR=${PROJECT_DIR}/resources \
python3.5 ${SCRIPT_TO_RUN} "$@"

deactivate
