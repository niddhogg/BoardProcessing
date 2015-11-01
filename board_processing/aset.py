"""
Author: Bogdan Kulynych (hello@bogdankulynych.me)
"""

import functools


def symmetric_memoization(fn):
    cache = {}

    def wrapped(a, b):
        if (a, b) in cache:
            return cache[(a, b)]
        if (b, a) in cache:
            return cache[(b, a)]
        else:
            result = fn(a, b)
            cache[(a, b)] = result
            return result
    return wrapped


class ApproximateSet(object):

    def __init__(self, eps, metric=lambda x, y: abs(x - y),
                 memoize=False):
        self.eps = eps
        self.centroids = []
        self.clusters = []

        self.metric = metric
        if memoize:
            self.metric = symmetric_memoization(self.metric)

    def _find_centroid_index(self, item):
        for index, centroid in enumerate(self.centroids):
            if self.metric(item, centroid) <= self.eps:
                return index
        return None

    def add(self, item):
        index = self._find_centroid_index(item)
        if index is None:
            self.centroids.append(item)
            self.clusters.append([item])
        else:
            def total_distance(candidate, cluster):
                return sum(self.metric(neighbour, candidate)
                           for neighbour in cluster)

            cluster = self.clusters[index]
            cluster.append(item)
            centroid = min(cluster, key=functools.partial(total_distance,
                                                          cluster=cluster))
            self.centroids[index] = centroid

    def __contains__(self, item):
        return self._find_centroid_index(item) is not None

    def __iter__(self):
        return self.centroids.__iter__()
