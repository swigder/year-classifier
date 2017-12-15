from collections import defaultdict

import itertools
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from files_to_data import read_data
from model import Model
from sklearn.feature_extraction.text import CountVectorizer


def plot_results_multi(multi_results, title, xlabel, ylabel='Validation Set Accuracy', legend_title=None):
    for label, results in multi_results.items():
        x = sorted(results.keys())
        plt.xticks(range(len(x)), x)
        y = [results[key] for key in x]
        plt.plot(range(len(y)), y, label=label)
    plt.legend(title=legend_title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_results(results, title, xlabel, ylabel='Validation Set Accuracy'):
    x = sorted(results.keys())
    plt.xticks(range(len(x)), x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    y = [results[key] for key in x]
    plt.plot(range(len(y)), y)
    # else:
    #     y1 = [results[key][0] for key in x]
    #     plt.plot(range(len(y1)), y1, color=y1_color)
    #     ax2 = plt.gca().twinx()
    #     y2 = [results[key][1] for key in x]
    #     ax2.plot(range(len(y2)), y2, color=y2_color)
    #     ax2.set_ylabel(ylabel_2, color=y2_color)
    #     ax2.tick_params('y', colors=y2_color)


def _read_data(data_dir, args={}):
    return read_data(data_dir, **args, reusable_input=True, validation_set=True)


def generate_min_sample_size_report(data_dir):
    sample_sizes = [100, 200, 500, 1000, 2000, 3000]

    print('Reading data...')
    inputs = dict()
    for min_sample in sample_sizes:
        inputs[min_sample] = _read_data(data_dir, {'min_sample_size': min_sample})

    print('Analyzing vocabulary...')
    vocab_size = dict()
    for min_sample in sample_sizes:
        print('Getting vocabulary size for sample size {}'.format(min_sample))
        data = inputs[min_sample]
        cv = CountVectorizer(max_df=.95, min_df=.0001, token_pattern=r"(?u)\b[A-ZÅÄÖa-zåäö][A-ZÅÄÖa-zåäö]+\b")
        cv.fit(data.train.inputs, data.train.targets)
        non_empty_train = 0
        for sample in data.train.inputs:
            if any(token in cv.vocabulary_ for token in cv.build_tokenizer()(sample)):
                non_empty_train += 1
        non_empty_test = 0
        for sample in data.test.inputs:
            if any(token in cv.vocabulary_ for token in cv.build_tokenizer()(sample)):
                non_empty_test += 1
        vocab_size[min_sample] = (len(cv.vocabulary_), len(data.train.targets), len(data.test.targets),
                                  non_empty_train / len(data.train.targets),
                                  non_empty_test / len(data.test.targets))
    for k, v in vocab_size.items():
        print(k, v)

    print('Testing accuracy...')
    accuracy_results = pd.DataFrame(columns=['Model', 'Min Sample', 'Accuracy'])
    for min_sample in sample_sizes:
        data = inputs[min_sample]
        for model_type in [Model.SGD_CLASSIFER, Model.NAIVE_BAYES, Model.MLP_CLASSIFER]:
            print('Running model {} for sample size {}'.format(model_type, min_sample))
            model = Model(model_type=model_type, verbose=False)
            model.train(data.train)
            accuracy = model.test(data.test)
            accuracy_results.loc[len(accuracy_results)] = [model_type, min_sample, accuracy]
            print('Got accuracy {}'.format(accuracy))
    print(accuracy_results.sort_values(['Model', 'Min Sample']))


def generate_feature_type_report(data_dir):
    print('Reading data...')
    data = _read_data(data_dir, {'min_sample_size': 1000})

    model_types = (Model.NAIVE_BAYES, Model.SGD_CLASSIFER, Model.MLP_CLASSIFER)
    feature_types = {(True, True): 'Boolean * IDF',
                     (True, False): 'Boolean',
                     (False, True): 'Tf-idf',
                     (False, False): 'Count'}
    accuracies = pd.DataFrame(columns=['Feature Type', *model_types])
    accuracies.set_index('Feature Type', inplace=True)

    print('Calculating accuracies...')
    for model_type in model_types:
        for (binary, tf_idf), feature_description in feature_types.items():
            vocab_options = {'binary': binary, 'use_tf_idf': tf_idf}
            model = Model(model_type=model_type, verbose=False, vocab_options=vocab_options)
            model.train(data.train)
            accuracy = model.test(data.test)
            accuracies.at[feature_description, model_type] = accuracy
            print(feature_description, model_type, accuracy)

    print(accuracies)


def generate_bayes_report(data_dir):
    sample_sizes = [100, 500, 1000]

    print('Reading data...')
    inputs = dict()
    for min_sample in sample_sizes:
        inputs[min_sample] = _read_data(data_dir, {'min_sample_size': min_sample})

    accuracies = defaultdict(dict)

    for min_sample in sample_sizes:
        for alpha in [0, .01, .1, .25, .5, 1.0]:
            data = inputs[min_sample]
            model = Model(model_type=Model.NAIVE_BAYES, verbose=False, model_options={'alpha': alpha})
            model.train(data.train)
            accuracy = model.test(data.test)
            accuracies[min_sample][alpha] = accuracy
    print(accuracies)
    plot_results_multi(accuracies,
                       title='Impact of smoothing on Naive Bayes Classifer',
                       xlabel='Alpha',
                       legend_title='Min sample size')


def generate_mlp_report(data_dir):
    data = _read_data(data_dir)
    results = dict()
    for hidden in [5, 10, 15, 20, 25, 50, 100, 200, 500, 1000]:
        model = Model(model_type=Model.MLP_CLASSIFER, verbose=False, model_options={'hidden_nodes': hidden})
        model.train(data.train)
        accuracy = model.test(data.test)
        results[hidden] = accuracy
        print(hidden, accuracy)
    print(results)
    plot_results(results, title='Impact of number of hidden nodes on MLP', xlabel='Number of hidden nodes')


def generate_sgd_report(data_dir):
    data = _read_data(data_dir)
    results = dict()
    for regularization in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        model = Model(model_type=Model.SGD_CLASSIFER, verbose=False, model_options={'regularization': regularization})
        model.train(data.train)
        accuracy = model.test(data.test)
        results[regularization] = accuracy
        print(regularization, accuracy)
    print(results)
    plot_results(results, title='Impact of regularization on logistic regression', xlabel='Regularization multiplier')


def generate_tf_idf_report(data_dir):
    sample_sizes = [100, 500, 1000, 2000]

    print('Reading data...')
    inputs = dict()
    for min_sample in sample_sizes:
        inputs[min_sample] = _read_data(data_dir, {'min_sample_size': min_sample})

    accuracies = defaultdict(dict)
    vocabulary_size = defaultdict(dict)
    for min_sample in sample_sizes:
        for min_df in [.005, .001, .0005, .0001, .00005, .00001]:
            data = inputs[min_sample]
            cv = CountVectorizer(min_df=min_df, token_pattern=r"(?u)\b[A-ZÅÄÖa-zåäö][A-ZÅÄÖa-zåäö]+\b")
            cv.fit(data.train.inputs, data.train.targets)

            model = Model(model_type=Model.NAIVE_BAYES, verbose=False,
                          vocab_options={'min_df': min_df})
            model.train(data.train)
            accuracy = model.test(data.test)
            print(min_sample, min_df, accuracy, len(cv.vocabulary_))
            accuracies[min_sample][min_df] = accuracy
            vocabulary_size[min_sample][min_df] = len(cv.vocabulary_)
    print(accuracies)
    print(vocabulary_size)
    plot_results_multi(accuracies,
                       title='Impact of min df on accuracy for different minimum sample size',
                       xlabel='Min document frequency for word in vocabulary',
                       legend_title='Min sample size')
    plt.figure()
    plot_results_multi(vocabulary_size,
                       title='Impact of min df on vocabulary size for different minimum sample size',
                       xlabel='Min document frequency for word in vocabulary',
                       ylabel='Vocabulary size',
                       legend_title='Min sample size')


REPORTS = {
    'min': generate_min_sample_size_report,
    'bayes': generate_bayes_report,
    'mlp': generate_mlp_report,
    'sgd': generate_sgd_report,
    'df': generate_tf_idf_report,
    'ft': generate_feature_type_report
}


def generate_report(data_dir, type):
    sns.set()
    REPORTS[type](data_dir)
    plt.show()
