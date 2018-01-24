from abc import abstractmethod

import numpy as np


class TestDataStatistic:
    def add_sents(self, sents):
        assert len(sents) == 2
        self.add_sents_pair(sents)

    @abstractmethod
    def add_sents_pair(self, sents):
        pass

    def add_all_sents(self, sents_list):
        for sents in sents_list:
            self.add_sents(sents)

    @abstractmethod
    def get_statistic(self):
        pass

    def evaluate(self, sents_list):
        self.add_all_sents(sents_list)
        return self.get_statistic()

    @staticmethod
    @abstractmethod
    def name():
        pass


class LengthStatistic(TestDataStatistic):
    def __init__(self, agg_func):
        self.result = []
        self.agg_func = agg_func

    def add_sents_pair(self, sents):
        self.result.extend([len(sents[0]), len(sents[1])])

    def get_statistic(self):
        return self.agg_func(np.array(self.result))

    @staticmethod
    def name():
        return "Sentence length"


class IntersectionStatistic(TestDataStatistic):
    def __init__(self, agg_func):
        self.result = []
        self.agg_func = agg_func

    def add_sents_pair(self, sents):
        sets = [set(s) for s in sents]
        self.result.append(len(sets[0] & sets[1]) / len(sets[0] | sets[1]))

    def get_statistic(self):
        return self.agg_func(np.array(self.result))

    @staticmethod
    def name():
        return "Sentences intersection"
