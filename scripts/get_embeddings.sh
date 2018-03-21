#!/bin/bash

set -e # stop on any error

cd "$(dirname "$0")"

START_DIR=$(pwd)
GLOVE_DIR="../resources/embeddings/glove"
BASE_URL="http://nlp.stanford.edu/data"

N_PACKETS=2
ZIP_FILES=("glove.840B.300d.zip" "glove.6B.zip")
PACKET_0_FILES=("glove.840B.300d.txt")
PACKET_1_FILES=("glove.6B.50d.txt" "glove.6B.100d.txt" "glove.6B.200d.txt" "glove.6B.300d.txt")

is_packet_present() {
    files=PACKET_$1_FILES[@]
    for file in ${!files}; do
        path="${START_DIR}/${GLOVE_DIR}/${file}"
        if [ ! -e ${path} ]; then
            return 1 # false
        fi
    done
    return 0 # true
}

echo "Checking for embeddings..."

mkdir -p "${GLOVE_DIR}" # create if not exists
cd ${GLOVE_DIR}

for ((i = 0; i < N_PACKETS; i++)); do
    tmp=ZIP_FILES[${i}]
    zip_file="${!tmp}"

    if is_packet_present ${i}; then
        echo "  Embeddings from ${zip_file} found."
    else
        echo "  Embeddings from ${zip_file} NOT found."

        # Prepare zip file
        if [ -e "${zip_file}" ]; then
            echo "    Zip file ${zip_file} found."
        else
            echo "    Zip file ${zip_file} not found."
            echo "      Downloading..."
            curl -LO "${BASE_URL}/${zip_file}"
            echo "      ...downloading done."
        fi

        # Extract
        echo "  Extracting ${zip_file} ..."
        unzip -o ${zip_file} # overwrite files if such exist
        echo "  ...extracting done."
    fi
done

echo "...embeddings are present."
