import numpy as np


class MeanMetrics:

    def __init__(self):
        self._figures = []

    def add(self, value):
        self._figures.append([value])

    def mean(self):
        return np.mean(self._figures)

    def median(self):
        return np.median(self._figures)

    def get_count(self):
        return len(self._figures)

    def get_sum(self):
        return np.sum(self._figures)

    def reset(self):
        self._figures = []
