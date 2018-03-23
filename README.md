# Building and clustering sentence embeddings

There are two main bash scripts of project:
1. scripts/run_docker.sh -- supports all options decribed below, but requires permissions to run docker,
2. scripts/run_virtualenv.sh -- requires less permissions, but has less features, e.g:
   * supports NLTK tokenizer only,
   * does not support automatic download of embeddings.

## Run modes

Following modes are avaliable to run script:
1. STS (default) -- evaluates given algorithm on Semantic Textual Similarity task,
2. stats -- generates statistics about STS datasets,
3. test -- runs smoke test of all algorithms and tokenizers,
4. train_s2s -- trains Seq2Seq model saved on disk.

## Algorithms

Following algorithms are available for evaluation
1. Autoencoder,
2. Doc2Vec,
3. GloveMean,
4. Seq2Seq,
5. SVD.

## Tokenizers

Following tokenizers are available to use during evaluation:
1. NLTK (default) -- tokenizer from NLTK library
2. Stanford -- tokenizer from Stanford CoreNLP library

## Sample commands
Command below are presented for scripts/run_docker.sh, but analogous parameters could be passed to
scripts/run_virtualenv.sh.

Evaluate algorithm Doc2Vec using default run mode with default tokenizer.
```
$ scripts/run_evaluation.sh Doc2Vec
```

Evaluate algorithm Doc2Vec on STS task (explicit ```-r STS``` is not necessary here) with Stanford tokenizer.
```
$ scripts/run_evaluation.sh Doc2Vec -r STS -t Stanford --alg-kwargs='{"vector_dim": 20, "epochs": 5}'
```
Doc2Vec object will be constructed as follows: ```Doc2Vec(vector_dim=20, epochs=5)```.
```--alg-kwargs``` parameter has to be in JSON format.

Print statistics about STS datasets
```
$ scripts/run_evaluation.sh -r stats
```

Run test of evaluation on tiny training set for all algorithms and tokenizers.
```
$ scripts/run_evaluation.sh -r test
```
This command tests only successful termination - it does not any checks of correctness.

Train Seq2Seq model - abort if model not exist.
```
$ scripts/run_evaluation.sh -r train_s2s
```
Train Seq2Seq model - create if model not exist.
```
$ scripts/run_evaluation.sh -r train_s2s --alg-kwargs='{"force_load": false}'
```
