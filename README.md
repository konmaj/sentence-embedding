# Building and clustering sentence embeddings

## Overview

In order to configure your system to run project, follow guidelines in these files:
1. [README_docker](./doc/README_docker.md) 
2. [README_GPU](./doc/README_GPU.md) &ndash; only if you want to use GPU.

There are two main bash scripts of project:
1. scripts/run_docker.sh
   * requires permissions to run docker,
   * supports all options described below,
   * does not contain tensorflow-gpu.
2. scripts/run_virtualenv.sh
   * requires less permissions
   * supports NLTK tokenizer only,
   * does not support automatic download of embeddings,
   * contains tensorflow-gpu &ndash; requires configured GPU to use tensorflow
   (see [README_GPU](./doc/README_GPU.md)).

## Run modes

Following modes are avaliable to run script:
1. STS (default) &ndash; evaluates given algorithm on Semantic Textual Similarity task,
2. stats &ndash; generates statistics about STS datasets,
3. test &ndash; runs smoke test of all algorithms and tokenizers,
4. get_resources &ndash; downloads and pre-computes all necessary resources for all algorithms and tokenizers.
5. train_s2s &ndash; trains given Seq2Seq model saved on disk.

## Algorithms

Following algorithms are available for evaluation
1. Autoencoder,
2. Doc2Vec,
3. FastTextMean,
4. FastTextMeanWithoutUnknown,
5. FastTextSVD,
6. FastTextSVDWithoutUnknown,
7. GloveMean,
8. GloveMeanNormalized,
9. S2SAutoencoder,
10. S2SAutoencoderWithCosine,
11. SVD,
12. GlovePosMean.

## Tokenizers

Following tokenizers are available to use during evaluation:
1. NLTK (default) &ndash; tokenizer from NLTK library
2. Stanford &ndash; tokenizer from Stanford CoreNLP library

## Sample commands
Command below are presented for scripts/run_docker.sh, but analogous parameters could be passed to
scripts/run_virtualenv.sh.

Evaluate algorithm Doc2Vec using default run mode with default tokenizer.
```
$ scripts/run_docker.sh Doc2Vec
```

Evaluate algorithm Doc2Vec on STS task (explicit ```-r STS``` is not necessary here) with Stanford tokenizer.
```
$ scripts/run_docker.sh Doc2Vec -r STS -t Stanford --alg-kwargs='{"vector_dim": 20, "epochs": 5}'
```
Doc2Vec object will be constructed as follows: ```Doc2Vec(vector_dim=20, epochs=5)```.
```--alg-kwargs``` parameter has to be in JSON format.

Print statistics about STS datasets
```
$ scripts/run_docker.sh -r stats
```

Run test of evaluation on tiny training set for all tokenizers and selected algorithms.
```
$ scripts/run_docker.sh -r test
```
This command tests only successful termination &ndash; it does not any checks of correctness.

Downloads and prepares resources for all algorithms (takes a really long time).
```
$ scripts/run_docker.sh -r get_resources
```

Train S2SAutoencoderWithCosine model &ndash; abort if model not exist.
```
$ scripts/run_docker.sh -r train_s2s S2SAutoencoderWithCosine
```
Train S2SAutoencoder model &ndash; create if model not exist. Note that ```false``` should be lowercase
and ```force_load``` should be surrounded by double quotation marks &ndash; requirements of JSON.
```
$ scripts/run_docker.sh -r train_s2s S2SAutoencoder --alg-kwargs='{"force_load": false}'
```
