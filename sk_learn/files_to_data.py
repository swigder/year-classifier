import codecs
import itertools
from collections import defaultdict
from collections import namedtuple

import os


Data = namedtuple('Data', ['train', 'test'])
DataSet = namedtuple('DataSet', ['inputs', 'targets'])


def data_set(data, reusable=False):
    return DataSet(inputs=inputs(data, reusable), targets=targets(data))


def inputs(data, reusable=False):
    iterable = itertools.chain.from_iterable(data.values())
    return iterable if not reusable else list(iterable)


def targets(data):
        return list(itertools.chain.from_iterable([[target] * len(items) for target, items in data.items()]))


def read_dir_into_dict(data_dir, max_samples_per_period=None, min_sample_size=None):
    data_dict = defaultdict(list)
    for filename in os.listdir(data_dir):
        if not filename.endswith('.txt'):
            continue
        with codecs.open(data_dir + filename, 'r', 'utf-8') as file:
            year = int(file.readline())
            # if year < 1780:
            #     continue
            lines = []
            current_line = ""
            for line in file.readlines():
                if max_samples_per_period and len(lines) > max_samples_per_period:
                    break
                if min_sample_size is not None and len(current_line) < min_sample_size and len(line) < min_sample_size:
                    current_line += line
                else:
                    lines.append(current_line)
                    current_line = line
            lines.append(current_line)
            data_dict[year] = lines
    return data_dict


def read_data(data_dir, max_samples_per_period, min_sample_size=None, reusable_input=False):
    args = {'max_samples_per_period': max_samples_per_period, 'min_sample_size': min_sample_size}
    train = read_dir_into_dict(data_dir + '/training/', **args)
    test = read_dir_into_dict(data_dir + '/test/', **args)

    return Data(train=data_set(train, reusable_input), test=data_set(test, reusable_input))
