from sent_emb.statistics.test_data import LengthStatistic, IntersectionStatistic, LengthDifferenceStatistic, \
                                          GloveCoverStatistic, CountSentencesStatistic
from sent_emb.evaluation.sts_read import TEST_NAMES, get_sts_input_path, read_sts_input_with_gs

import numpy as np


AGG_FUNCS = [[np.min, 'min'], [np.max, 'max'], [np.mean, 'mean'], [np.std, 'std']]
STATISTICS = [LengthStatistic, LengthDifferenceStatistic, IntersectionStatistic]
SINGLE_STATISTICS = [GloveCoverStatistic, CountSentencesStatistic]


def test_data_statistics(sents_pairs, data_name):
    print(data_name + ': ')
    for stat_class in STATISTICS:
        print('  {}:'.format(stat_class.name()))

        for func, func_name in AGG_FUNCS:
            stat = stat_class(func)
            print('    {}: {}'.format(func_name, stat.evaluate(sents_pairs)))

    for stat_class in SINGLE_STATISTICS:
        print('  {}:'.format(stat_class.name()))

        stat = stat_class(np.mean)
        print('    {}: {}'.format('mean', stat.evaluate(sents_pairs)))
    print()


def all_statistics(tokenizer):
    for year, test_names in sorted(TEST_NAMES.items()):
        for test_name in test_names:
            input_path = get_sts_input_path(year, test_name)
            sents_pairs_gs = read_sts_input_with_gs(year, test_name, tokenizer)
            sents_pairs = [(s1, s2) for (s1, s2, gs) in sents_pairs_gs if gs != None]

            test_data_statistics(sents_pairs, 'STS{}, {}'.format(year, test_name))
