from sent_emb.statistics.test_data import LengthStatistic, IntersectionStatistic
from sent_emb.evaluation.sts import get_sts_input_path, read_sts_input, TEST_NAMES

import numpy as np


AGG_FUNCS = [[np.min, 'min'], [np.max, 'max'], [np.mean, 'mean'], [np.std, 'std']]
STATISTICS = [LengthStatistic, IntersectionStatistic]


def test_data_statistics(sents_pairs, data_name):
    print(data_name + ': ')
    for stat_class in STATISTICS:
        print('  {}:'.format(stat_class.name()))

        for func, func_name in AGG_FUNCS:
            stat = stat_class(func)
            print('    {}: {}'.format(func_name, stat.evaluate(sents_pairs)))
        print()
    print()


def all_statistics(tokenizer):
    for year, test_names in sorted(TEST_NAMES.items()):
        for test_name in test_names:
            input_path = get_sts_input_path(year, test_name)
            all_sents = read_sts_input(input_path, tokenizer)

            assert len(all_sents) % 2 == 0

            sents_pairs = list(zip(*[all_sents[i::2] for i in range(0, 2)]))

            test_data_statistics(sents_pairs, 'STS{}, {}'.format(year, test_name))
