# coding=utf-8
import argparse

import time

from files_to_data import read_data
from model import Model
from report import generate_report

MAX_SAMPLES_PER_PERIOD = 5000
PERIOD_LENGTH = 20
MIN_SAMPLE_SIZE = None
DEFAULT_MODEL_TYPE = Model.MLP_CLASSIFER


parser = argparse.ArgumentParser(description='Build a model.')
parser.add_argument('dir', type=str, help='directory with input and output subdirectories')
parser.add_argument('-m', '--max_samples', type=int, default=MAX_SAMPLES_PER_PERIOD,
                    help='max number of samples per period')
parser.add_argument('-s', '--min_sample', type=int, default=MIN_SAMPLE_SIZE,
                    help='minimum number of characters per sample')
parser.add_argument('-t', '--model_type', default=DEFAULT_MODEL_TYPE,
                    choices=Model.MODEL_OPTIONS,
                    help='type of model')
parser.add_argument('-r', '--report', default=None, choices=['min', 'bayes', 'df'],
                    help='type of report')

args = parser.parse_args()

if args.report is None:
    start_time = time.time()
    data = read_data(args.dir, args.max_samples, args.min_sample)
    print('Read data in {:.2f} seconds.'.format(time.time() - start_time))

    model = Model(model_type=args.model_type)
    model.train(data.train)
    model.visualize()
    model.test(data.test)
else:
    generate_report(args.dir, args.report)
