import pandas as pd

from files_to_data import read_data
from model import Model
from sklearn.feature_extraction.text import CountVectorizer


def generate_min_sample_size_report(data_dir):
    sample_sizes = [100, 200, 500, 1000, 2000, 3000]

    print('Reading data...')
    inputs = dict()
    for min_sample in sample_sizes:
        inputs[min_sample] = read_data(data_dir, None, min_sample, reusable_input=True)

    print('Analyzing vocabulary...')
    vocab_size = dict()
    for min_sample in sample_sizes:
        print('Getting vocabulary size for sample size {}'.format(min_sample))
        data = inputs[min_sample]
        cv = CountVectorizer(max_df=.95, min_df=.0001, token_pattern=r"(?u)\b[A-ZÅÄÖa-zåäö][A-ZÅÄÖa-zåäö]+\b")
        cv.fit_transform(data.train.inputs, data.train.targets)
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
