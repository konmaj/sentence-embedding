import csv

from sent_emb.algorithms.path_utility import DATASETS_DIR
from sent_emb.evaluation.model import DataSet, flatten_sent_pairs, SentPair, zip_sent_pairs_with_gs


STS12_TRAIN_NAMES = ['MSRpar', 'MSRvid', 'SMTeuroparl']

TEST_NAMES = {
    12: ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews'],
    13: ['headlines', 'OnWN', 'FNWN'],
    14: ['deft-forum', 'deft-news', 'headlines', 'images', 'OnWN', 'tweet-news'],
    15: ['answers-forums', 'answers-students', 'belief', 'headlines', 'images'],
    16: ['answer-answer', 'headlines', 'plagiarism', 'postediting', 'question-question'],
}


class STS(DataSet):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def word_set(self):
        if not hasattr(self, '_word_set_value'):
            input_paths = [get_sts_input_path(12, train_name, use_train_set=True)
                           for train_name in STS12_TRAIN_NAMES]
            input_paths += [get_sts_input_path(year, test_name, use_train_set=False)
                            for year, test_names in sorted(TEST_NAMES.items())
                            for test_name in test_names]

            sts_words = set()
            for input_path in input_paths:
                sent_pairs = read_sts_input(input_path, self.tokenizer)
                sents = flatten_sent_pairs(sent_pairs)
                for sent in sents:
                    for word in sent:
                        sts_words.add(word)
            self._word_set_value = sts_words

        return self._word_set_value

    def tokenizer_name(self):
        return self.tokenizer.name()


def get_sts_path(year):
    assert year in TEST_NAMES
    return DATASETS_DIR.joinpath('STS{}'.format(year))


def get_sts_input_path(year, test_name, use_train_set=False):
    assert year in TEST_NAMES
    assert not (use_train_set and year != 12)

    sts_path = get_sts_path(year)
    dir_name = 'test-data' if not use_train_set else 'train-data'
    input_name = 'STS.input.{}.txt'.format(test_name)

    return sts_path.joinpath(dir_name, input_name)


def get_sts_gs_path(year, test_name, use_train_set=False):
    assert year in TEST_NAMES
    assert not (use_train_set and year != 12)

    sts_path = get_sts_path(year)
    dir_name = 'test-data' if not use_train_set else 'train-data'
    gs_name = 'STS.gs.{}.txt'.format(test_name)

    return sts_path.joinpath(dir_name, gs_name)


def get_sts_output_path(year, test_name):
    assert year in TEST_NAMES

    sts_path = get_sts_path(year)
    output_name = 'STS.output.{}.txt'.format(test_name)

    return sts_path.joinpath('out', output_name)


def tokens(tokenizer, sents):
    """
    Tokenizes each sentence in a list.

    :param tokenizer: tokenizer to use
    :param sents: list of sentences (list of strings) to tokenize
    :return: list of tokenized sentences - each sentence is represented as a list of words
    """
    guard = "verylongwordwhichisntawordanddoesntappearinlanguage"
    con = ''
    for sent in sents:
        con = con + sent + '\n' + guard + '\n'
    tokenized = tokenizer.tokenize(con)
    res = [[]]
    for word in tokenized:
        if word == guard:
            res.append([])
        else:
            res[-1].append(word)
    return res[:-1]


def read_sts_input(file_path, tokenizer):
    """
    Reads STS input file at given `file_path`.

    returns: list of SentPairs
    """
    sents = []
    with open(str(file_path), 'r') as test_file:
        test_reader = csv.reader(test_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in test_reader:
            assert len(row) == 2 \
                or len(row) == 4  # STS16 contains also source of each sentence
            sents.extend(row[:2])

    sents = tokens(tokenizer, sents)

    return [SentPair(sent1, sent2) for sent1, sent2 in zip(sents[::2], sents[1::2])]


def read_sts_gs(file_path):
    """
    Reads STS gold standard file at given `file_path`.

    :param file_path: Path to gold standard file
    :return: list of gold standard scores (floats) for pairs of sentences
        Missing scores are represented as `None` in the resulting list
        (some scores in STS15 and STS16 are missing).
    """
    gs_score = []
    with open(str(file_path), 'r') as gs_file:
        for line in gs_file.readlines():
            assert line[-1] == '\n'
            gs_score.append(float(line[:-1]) if len(line) > 1 else None)
    return gs_score


def read_sts_input_with_gs(year, test_name, tokenizer, use_train_set=False):
    """
    Reads STS input with gold standards for given test and year.

    :param year: two last digits of year of STS task (e.g. 12)
    :param test_name: string as in `TEST_NAMES` or `STS12_TRAIN_NAMES`
    :param tokenizer: tokenizer to use while reading sentences
    :param use_train_set: whether to search in `train-data` directory; otherwise in `test-data`.
    :return: list of SentPairWithGs objects
    """
    input_path = get_sts_input_path(year, test_name, use_train_set=use_train_set)
    sents_pairs = read_sts_input(input_path, tokenizer)

    gs_path = get_sts_gs_path(year, test_name, use_train_set=use_train_set)
    gold_standards = read_sts_gs(gs_path)

    return zip_sent_pairs_with_gs(sents_pairs, gold_standards)


def read_train_set(year, tokenizer):
    """
    Reads training set available for STS in given 'year'.

    For each year training set consists of:
    1) STS12 train-data
    2) Test data from STS from former years

    :param year: two last digits of year of STS task (e.g. 12)
    :param tokenizer: tokenizer to use while reading sentences
    :return: list of SentPairWithGs objects from all training sets available for STS
             in given `year`
    """
    # STS12 train-data...
    train_data = []
    for test_name in STS12_TRAIN_NAMES:
        train_data.extend(read_sts_input_with_gs(12, test_name, tokenizer, use_train_set=True))

    # test sets from STS before given 'year'
    for test_year, test_names_year in sorted(TEST_NAMES.items()):
        if test_year >= year:
            break
        for test_name in test_names_year:
            train_data.extend(read_sts_input_with_gs(test_year, test_name, tokenizer,
                                                     use_train_set=False))
    return train_data
