import codecs
import itertools
from collections import defaultdict
from collections import namedtuple

import os


Data = namedtuple('Data', ['train', 'test'])
DataSet = namedtuple('DataSet', ['inputs', 'targets'])


def data_set(data):
    return DataSet(inputs=inputs(data), targets=targets(data))


def inputs(data):
    return itertools.chain.from_iterable(data.values())


def targets(data):
        return list(itertools.chain.from_iterable([[target] * len(items) for target, items in data.items()]))


def read_data(data_dir, max_samples_per_period, period_length, filter=None):
    train = defaultdict(list)
    test = defaultdict(list)

    for filename in os.listdir(data_dir):
        if not filename.endswith('.txt'):
            continue
        with codecs.open(data_dir + filename, 'r', 'utf-8') as file:
            year = int(file.readline())
            year = year // period_length * period_length
            if filter:
                lines = [line for line in file.readlines() if (filter(line))]
            else:
                lines = list(file.readlines())
            split = int(len(lines) * .75)
            train[year] += lines[:split]
            test[year] += lines[split:]
            if len(train[year]) > max_samples_per_period:
                train[year] = train[year][:max_samples_per_period]
            if len(test[year]) > max_samples_per_period:
                test[year] = test[year][:max_samples_per_period]

    return Data(train=data_set(train), test=data_set(test))
