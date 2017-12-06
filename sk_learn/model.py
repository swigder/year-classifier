import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


class Model:
    def __init__(self):
        self.text_clf = Pipeline([('vect', CountVectorizer(max_df=.95, min_df=10)),
                                  ('tfidf', TfidfTransformer()),
                                  # ('clf', SGDClassifier(loss='squared_loss', penalty='l1',
                                  #                       alpha=1e-3, random_state=42,
                                  #                       max_iter=5, tol=None, class_weight='balanced',
                                  #                       verbose=3)),
                                  # ('clf', SGDRegressor(loss='squared_loss', penalty='l1',
                                  #                      alpha=1e-3, random_state=42,
                                  #                      max_iter=5, tol=None,
                                  #                      verbose=3)),
                                  ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5,
                                                        verbose=True))
                                  ])
        print(self.text_clf)

    def train(self, training):
        self.text_clf.fit(training.inputs, training.targets)

    def test(self, test):
        inputs = list(test.inputs)

        predicted = self.text_clf.predict(inputs)

        print(metrics.classification_report(test.targets, predicted))
        print(metrics.confusion_matrix(test.targets, predicted))

        # for i in range(len(inputs)):
        #     if abs(test.targets[i] - predicted[i]) > 100:
        #         print(test.targets[i], predicted[i], inputs[i])

    def visualize(self):
        print()
        v = list(self.text_clf.named_steps['vect'].vocabulary_.keys())
        # for target, coeffs in enumerate(self.text_clf.named_steps['clf'].coef_):
        #     top = np.argsort(coeffs)[-10:]
        #     print(u"{}\n{}".format(target,
        #                            u"\n".join([u"{} {}".format(v[i], coeffs[i]) for i in top])))
        for target, coeffs in enumerate(self.text_clf.named_steps['clf'].coefs_[0].T):
            top = np.argsort(coeffs)[-10:]
            print(u"{}\n{}".format(target,
                                   u"\n".join([u"{} {}".format(v[i], coeffs[i]) for i in top])))
        print()
