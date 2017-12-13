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

    def visualize(self):
        if self.model_type in [self.MLP_REGRESSOR, self.SGD_REGRESSOR]:
            return

        print()

        vocab = self.text_clf.named_steps['vect'].vocabulary_
        terms = np.array(list(vocab.keys()))
        indices = np.array(list(vocab.values()))
        v = terms[np.argsort(indices)]

        if self.model_type == self.SGD_CLASSIFER:
            for target, coeffs in enumerate(self.text_clf.named_steps['clf'].coef_):
                top = np.argsort(coeffs)[-10:]
                print(u"{}\n{}".format(self.text_clf.named_steps['clf'].classes_[target],
                                       u"\n".join([u"{} {}".format(v[i], coeffs[i]) for i in top])))
        elif self.model_type == self.MLP_CLASSIFER:
            # todo find hidden nodes with most variance
            # todo what do they say?
            layer0 = self.text_clf.named_steps['clf'].coefs_[0]
            layer1 = self.text_clf.named_steps['clf'].coefs_[1]
            classes = self.text_clf.named_steps['clf'].classes_
            vocab_var = np.var(self.text_clf.named_steps['clf'].coefs_[0], axis=1)
            n_words_to_examine = 50
            ind = np.argpartition(vocab_var, -n_words_to_examine)[-n_words_to_examine:]
            ind = ind[np.argsort(vocab_var[ind])][::-1]
            for i in ind:
                top_hidden = np.argmax(layer0[i])
                hidden_target = np.argmax(layer1[top_hidden])
                print(v[i], np.var(layer0[i]), classes[hidden_target], self.text_clf.predict([v[i]]))
        elif self.model_type == self.NAIVE_BAYES:
            import matplotlib.pyplot as plt
            import seaborn as sns

            classes = self.text_clf.named_steps['clf'].classes_
            coeffs = self.text_clf.named_steps['clf'].coef_
            vocab_var = np.var(coeffs, axis=0)
            n_words_to_examine = 100
            ind = np.argpartition(vocab_var, -n_words_to_examine)[-n_words_to_examine:]
            ind = ind[np.argsort(vocab_var[ind])][::-1]
            for i in ind:
                word_coeffs = coeffs[:,i]
                year, variance = classes[np.argmax(word_coeffs)], np.var(word_coeffs)
                plt.scatter(year, variance)
                plt.annotate(v[i], (year, variance))
                print(v[i], year, variance, coeffs)
            plt.show()
        print()
