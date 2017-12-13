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


def split_data_set(data_set, training=.9):
    cutoff = int(len(data_set.inputs) * training)
    return DataSet(inputs=data_set.inputs[:cutoff], targets=data_set.targets[:cutoff]), \
           DataSet(inputs=data_set.inputs[cutoff:], targets=data_set.targets[cutoff:])


def read_dir_into_data_set(data_dir, reusable=False, max_samples_per_period=None, min_sample_size=None):
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
    return data_set(data_dict, reusable)


def read_data(data_dir, max_samples_per_period=None, min_sample_size=None, reusable_input=False, validation_set=False):
    args = {'max_samples_per_period': max_samples_per_period,
            'min_sample_size': min_sample_size,
            'reusable': reusable_input}

    if not validation_set:
        train = read_dir_into_data_set(data_dir + '/training/', **args)
        test = read_dir_into_data_set(data_dir + '/test/', **args)
        return Data(train=train, test=test)
    else:
        args['reusable'] = True
        full_data_set = read_dir_into_data_set(data_dir + '/training/', **args)
        train, test = split_data_set(full_data_set)
        return Data(train=train, test=test)
