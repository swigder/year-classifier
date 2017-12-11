import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline


class Model:
    SGD_CLASSIFER = 'sgd'
    MLP_CLASSIFER = 'mlp'
    SGD_REGRESSOR = 'sgd_r'
    MLP_REGRESSOR = 'mlp_r'
    MODEL_OPTIONS = [SGD_CLASSIFER, MLP_CLASSIFER, SGD_REGRESSOR, MLP_REGRESSOR]

    def __init__(self, model_type, max_df=.95, min_df=.0001, binary=True):
        self.model_type = model_type
        self.text_clf = Pipeline([('vect', CountVectorizer(max_df=max_df, min_df=min_df, binary=binary,
                                                           token_pattern=r"(?u)\b[A-ZÅÄÖa-zåäö][A-ZÅÄÖa-zåäö]+\b")),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', self.get_model(model_type))
                                  ])
        for name, step in self.text_clf.named_steps.items():
            print(name, step)

    def get_model(self, model_type, hidden_layers=100):
        if model_type == self.SGD_CLASSIFER:
            return SGDClassifier(loss='modified_huber', penalty='l1',
                                 alpha=1e-3, random_state=42,
                                 max_iter=30, tol=None, class_weight='balanced',
                                 verbose=3)
        elif model_type == self.MLP_CLASSIFER:
            return MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(hidden_layers,),
                                 learning_rate_init=1e-1, learning_rate='adaptive',
                                 validation_fraction=.2,
                                 verbose=True)
        elif model_type == self.SGD_REGRESSOR:
            return SGDRegressor(loss='squared_loss', penalty='l1',
                                alpha=1e-3, random_state=42,
                                max_iter=5, tol=None,
                                verbose=3)
        elif model_type == self.MLP_REGRESSOR:
            return MLPRegressor(verbose=True, learning_rate_init=1e-1, learning_rate='adaptive')

    def train(self, training):
        self.text_clf.fit(training.inputs, training.targets)

        v = self.text_clf.named_steps['vect']
        print()
        print('Vocabulary size: {} ({} words removed)'.format(len(v.vocabulary_), len(v.stop_words_)))

    def test(self, test):
        inputs = list(test.inputs)

        predicted = self.text_clf.predict(inputs)

        if self.model_type in [self.MLP_REGRESSOR, self.SGD_REGRESSOR]:
            target_options = list(sorted(set(test.targets)))
            predicted = [target_options[np.argmin([abs(p - t) for t in target_options])] for p in predicted]

        df = pd.DataFrame(data={'actual': test.targets, 'predicted': predicted})
        # for k, v in df.groupby([df.actual, df.predicted]).groups.items():
        #     print(k, inputs[v[0]])
        correct_count = df.where(df.actual == df.predicted).count()[0]
        almost_correct_count = correct_count + df.where(abs(df.actual - df.predicted) == 20).count()[0]

        print()
        print('Accuracy: {:.4f} ({} / {})\n'
              .format(correct_count / len(inputs), correct_count, len(inputs)))
        print('Accuracy (within adjoining period): {:.4f} ({} / {})\n'
              .format(almost_correct_count / len(inputs), almost_correct_count, len(inputs)))
        print(metrics.classification_report(test.targets, predicted))
        print(metrics.confusion_matrix(test.targets, predicted, list(sorted(set(test.targets)))))

    def visualize(self):
        if self.model_type in [self.MLP_REGRESSOR, self.SGD_REGRESSOR]:
            return

        print()

        v = list(self.text_clf.named_steps['vect'].vocabulary_.keys())

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

        print()
