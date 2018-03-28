# Building and clustering sentence embeddings

The main script of project is scripts/run_evaluation.sh.

## Run modes

Following modes are avaliable to run script:
1. STS (default) - evaluates given algorithm on Semantic Textual Similarity task,
2. stats - generates statistics about STS datasets,
3. test - runs smoke test of all algorithms and tokenizers,
4. train_s2s - trains Seq2Seq model saved on disk.

## Algorithms

Following algorithms are available for evaluation
1. Doc2Vec,
2. Seq2Seq.

Other algorithms will be added after refactoring.

## Tokenizers

Following tokenizers are available to use during evaluation:
1. NLTK (default) - tokenizer from NLTK library
2. Stanford - tokenizer from Stanford CoreNLP library

## Sample commands
Evaluate algorithm Doc2Vec using default run mode with default tokenizer.
```
$ scripts/run_evaluation.sh Doc2Vec
```

Evaluate algorithm Doc2Vec on STS task (explicit ```-r STS``` is not necessary here) with Stanford tokenizer.
```
$ scripts/run_evaluation.sh Doc2Vec -r STS -t Stanford --alg-kwargs '{"vector_dim": 20, "epochs": 5}'
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
$ scripts/run_evaluation.sh -r train_s2s --alg-kwargs '{"force_load": false}'
```
