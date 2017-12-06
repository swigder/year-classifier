import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


class Model:
    def __init__(self):
        self.text_clf = Pipeline([('vect', CountVectorizer(max_df=.9, min_df=5)),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', SGDClassifier(loss='squared_loss', penalty='l1',
                                                        alpha=1e-3, random_state=42,
                                                        max_iter=5, tol=None, class_weight='balanced',
                                                        verbose=3)),
                                  # ('clf', SGDRegressor(loss='squared_loss', penalty='l1',
                                  #                      alpha=1e-3, random_state=42,
                                  #                      max_iter=5, tol=None,
                                  #                      verbose=3)),
                                  ])
        print(self.text_clf)

    def train(self, training):
        self.text_clf.fit(training.inputs, training.targets)

    def test(self, test):
        predicted = self.text_clf.predict(test.inputs)

        print(metrics.classification_report(test.targets, predicted))
        print(metrics.confusion_matrix(test.targets, predicted))

    def visualize(self):
        v = list(self.text_clf.named_steps['vect'].vocabulary_.keys())
        for target, coeffs in enumerate(self.text_clf.named_steps['clf'].coef_):
            top_10 = np.argsort(np.abs(coeffs))[-5:]
            print(u"{}\n{}".format(target, u"\n".join([u"{} {}".format(v[i], coeffs[i]) for i in top_10])))
