from abc import ABC, abstractmethod

import numpy as np

from sent_emb.algorithms.glove_utility import GLOVE_FILE, read_file


class TestDataStatistic(ABC):
    def __init__(self, agg_func):
        self.results = []
        self.agg_func = agg_func

    def add_sents(self, sents):
        assert len(sents) == 2
        self.add_sents_pair(sents)

    @abstractmethod
    def add_sents_pair(self, sents):
        pass

    def add_all_sents(self, sents_list):
        for sents in sents_list:
            self.add_sents(sents)

    def get_statistic(self):
        return self.agg_func(np.array(self.results))

    def evaluate(self, sents_list):
        self.results = []
        self.add_all_sents(sents_list)
        return self.get_statistic()

    @staticmethod
    @abstractmethod
    def name():
        pass


class CountSentencesStatistic(TestDataStatistic):
    def __init__(self, agg_func):
        super().__init__(agg_func)
        self.agg_func = np.sum

    def add_sents_pair(self, sents):
        self.results.append([2])

    @staticmethod
    def name():
        return "Number of sentences"


class LengthStatistic(TestDataStatistic):
    def __init__(self, agg_func):
        super().__init__(agg_func)

    def add_sents_pair(self, sents):
        self.results.extend([len(sents[0]), len(sents[1])])

    @staticmethod
    def name():
        return "Sentence length"


class LengthDifferenceStatistic(TestDataStatistic):
    def __init__(self, agg_func):
        super().__init__(agg_func)

    def add_sents_pair(self, sents):
        self.results.append(abs(len(sents[0]) - len(sents[1])))

    @staticmethod
    def name():
        return "Sentences length difference"


class IntersectionStatistic(TestDataStatistic):
    def __init__(self, agg_func):
        super().__init__(agg_func)

    def add_sents_pair(self, sents):
        sets = [set(s) for s in sents]
        self.results.append(len(sets[0] & sets[1]) / len(sets[0] | sets[1]))

    @staticmethod
    def name():
        return "Sentences intersection"


class GloveCoverStatistic(TestDataStatistic):
    def __init__(self, agg_func):
        super().__init__(agg_func)
        self.glove_words = set()

        def process(word, vec = None, raw_sent = None):
            self.glove_words.add(word)

        read_file(GLOVE_FILE, process)

    def add_sents_pair(self, sents):
        for sent in sents:
            cover_count = 0
            for word in sent:
                if word in self.glove_words:
                    cover_count += 1

            self.results.append(cover_count / len(sent))

    @staticmethod
    def name():
        return "GloVe coverage"
