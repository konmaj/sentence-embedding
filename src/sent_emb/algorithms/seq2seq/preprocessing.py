import math
import numpy as np

from sent_emb.evaluation.model import flatten_sent_pairs


def replace_with_embs(sents, word_embedding):
    """
    Converts sentences to lists of their word embeddings.

    sents: list of tokenized sentences - each sentence is a list of strings

    word_embedding: source of word embeddings (object of WordEmbedding class)

    returns: list of sentences
        sentence: list of embeddings
        embedding: list of floats
    """
    word_vec_dict = word_embedding.embeddings(sents)

    sents_vec = []
    for sent in sents:
        cur_sent = []
        for word in sent:
            cur_sent.append(word_vec_dict[word])
        sents_vec.append(cur_sent)

    return sents_vec


def get_random_subsequence(sequence, result_size):
    """
    Computes random subsequence of size `result_size` of given `sequence`.

    sequence: any data structure which supports indexing (e.g. list)
    result_size: size of resulting subsequence

    returns: subsequence of `sequence` in form of list
    """
    seq_len = len(sequence)
    assert result_size <= seq_len

    selected_indices = np.sort(np.random.permutation(seq_len)[: result_size])

    return [sequence[ind] for ind in np.nditer(selected_indices)]


def align_sents(sents_vec, padding_vec, cut_rate=0.8):
    """
    Fits each sentence to has equal number of words (depending on `cut_rate`).

    sents_vec: list of sentences of vectorized words
               (see return type of `replace_with_embs` function)

    padding_vec: `np.array` of type `np.float` and length `GLOVE_DIM`
                 is used when there is not enough words in the sentence.

    cut_rate: coefficient of [0; 1] interval
              Target number of words per sentence (`num_encoder_words`) is set to be the minimal
              integer such that at least `cut_rate` fraction of original sentences are of length
              less or equal `num_encoder_words`.

    returns: list of sentences (in format as `sents_vec`)
             each sentence consists of MAX_ENCODER_WORDS words.
    """
    assert 0 <= cut_rate <= 1

    # sents_vec.sort(key = lambda s: len(s))

    sent_lengths = sorted([len(sent) for sent in sents_vec])
    num_encoder_words = sent_lengths[int(math.ceil(cut_rate * len(sent_lengths)))]

    for i in range(len(sents_vec)):
        if len(sents_vec[i]) <= num_encoder_words:
            sents_vec[i].extend([padding_vec for _ in range(num_encoder_words - len(sents_vec[i]))])
        else:
            sents_vec[i] = get_random_subsequence(sents_vec[i], num_encoder_words)
        assert len(sents_vec[i]) == num_encoder_words

    return sents_vec


def preprocess_sents(sents, word_embedding, rand=0.0):
    """
    Prepares sentences to be put into Seq2Seq neural net.

    sents: list of tokenized sentences - each sentence is a list of strings

    word_embedding: source of word embeddings (object of WordEmbedding class)

    returns: numpy 3-D array of floats, which represents list of sentences of vectorized words
    """
    sents_vec = replace_with_embs(sents, word_embedding)
    # print(sents_vec[:5])
    # for si in range(len(sents_vec)):
    #     for wi in range(len(sents_vec[si])):
    #         sents_vec[si][wi] += np.random.rand(sents_vec[si][wi].shape[0]) * rand

    padding_vec = np.zeros(word_embedding.get_dim(), dtype=np.float)
    aligned_sents = align_sents(sents_vec, padding_vec)

    return np.array(aligned_sents, dtype=np.float)


def preprocess_sent_pairs(sent_pairs, word_embedding, rand=0.0):
    """
    Prepares pairs of sentences to be put into Seq2Seq neural net.

    sents: list of SentPair objects

    word_embedding: source of word embeddings (object of WordEmbedding class)

    returns: pair of numpy 3-D arrays of floats - each array as in `preprocess_sents` return value
    """
    sents = flatten_sent_pairs(sent_pairs)
    sents = preprocess_sents(sents, word_embedding, rand)

    first_sents = sents[0::2]
    second_sents = sents[1::2]

    assert first_sents.shape == second_sents.shape
    return first_sents, second_sents
