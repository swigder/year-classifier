import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline


class Model:
    SGD_CLASSIFER = 'sgd'
    MLP_CLASSIFER = 'mlp'
    SGD_REGRESSOR = 'sgd_r'
    MLP_REGRESSOR = 'mlp_r'
    NAIVE_BAYES = 'bayes'
    MODEL_OPTIONS = [SGD_CLASSIFER, MLP_CLASSIFER, SGD_REGRESSOR, MLP_REGRESSOR, NAIVE_BAYES]

    def __init__(self, model_type, verbose=True, vocab_options={}, model_options={}):
        self.model_type = model_type
        self.verbose = verbose

        self.text_clf = self.get_pipeline(**vocab_options, model_options=model_options)

    def get_pipeline(self, max_df=.95, min_df=.0001, use_tf_idf=True, binary=False, model_options={}):
        steps = list()

        steps.append(('vect', CountVectorizer(max_df=max_df, min_df=min_df, binary=binary, ngram_range=(1, 1),
                                              token_pattern=r"(?u)\b[A-ZÅÄÖa-zåäö][A-ZÅÄÖa-zåäö]+\b")))
        if use_tf_idf:
            steps.append(('tfidf', TfidfTransformer()))
        steps.append(('clf', self.get_model(self.model_type, **model_options)))

        if self.verbose:
            for name, step in steps:
                print(name, step)

        return Pipeline(steps)

    def get_model(self, model_type, hidden_nodes=50, alpha=.1, regularization=1e-5):
        if model_type == self.SGD_CLASSIFER:
            return SGDClassifier(loss='log', penalty='l2',
                                 alpha=regularization, random_state=42,
                                 max_iter=30, tol=None, class_weight='balanced',
                                 verbose=3 if self.verbose else 0)
        elif model_type == self.MLP_CLASSIFER:
            return MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(hidden_nodes,),
                                 learning_rate_init=1e-1, learning_rate='adaptive',
                                 validation_fraction=.2,
                                 verbose=self.verbose)
        elif model_type == self.SGD_REGRESSOR:
            return SGDRegressor(loss='squared_loss', penalty='l2',
                                alpha=1e-4, random_state=42,
                                max_iter=30, tol=None,
                                verbose=3 if self.verbose else 0)
        elif model_type == self.MLP_REGRESSOR:
            return MLPRegressor(verbose=self.verbose, learning_rate_init=1e-1, learning_rate='adaptive')
        elif model_type == self.NAIVE_BAYES:
            return MultinomialNB(fit_prior=False, alpha=alpha)

    def train(self, training):
        self.text_clf.fit(training.inputs, training.targets)

        v = self.text_clf.named_steps['vect']

        if self.verbose:
            print()
            print('Vocabulary size: {} ({} words removed)'.format(len(v.vocabulary_), len(v.stop_words_)))

    def test(self, test):
        inputs = list(test.inputs)

        predicted = self.text_clf.predict(inputs)

        if self.model_type in [self.MLP_REGRESSOR, self.SGD_REGRESSOR]:
            target_options = list(sorted(set(test.targets)))
            predicted = [target_options[np.argmin([abs(p - (t + 10)) for t in target_options])] for p in predicted]

        df = pd.DataFrame(data={'actual': test.targets, 'predicted': predicted})
        correct_count = df.where(df.actual == df.predicted).count()[0]
        almost_correct_count = correct_count + df.where(abs(df.actual - df.predicted) == 20).count()[0]

        if self.verbose:
            print()
            print('Accuracy: {:.4f} ({} / {})\n'
                  .format(correct_count / len(inputs), correct_count, len(inputs)))
            print('Accuracy (within adjoining period): {:.4f} ({} / {})\n'
                  .format(almost_correct_count / len(inputs), almost_correct_count, len(inputs)))
            print(metrics.classification_report(test.targets, predicted))
            print(metrics.confusion_matrix(test.targets, predicted, list(sorted(set(test.targets)))))

        return correct_count / len(inputs)
