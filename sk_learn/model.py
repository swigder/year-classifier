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
    MODEL_OPTIONS = [SGD_CLASSIFER, MLP_CLASSIFER, SGD_REGRESSOR]

    def __init__(self, model_type):
        self.model_type = model_type
        self.text_clf = Pipeline([('vect', CountVectorizer(max_df=.95, min_df=10)),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', self.get_model(model_type))
                                  # ('clf', MLPRegressor(solver='lbfgs', alpha=1e-5,
                                  #                      verbose=True))
                                  ])
        print(self.text_clf)

    def get_model(self, model_type):
        if model_type == self.SGD_CLASSIFER:
            return SGDClassifier(loss='squared_loss', penalty='l1',
                                 alpha=1e-3, random_state=42,
                                 max_iter=5, tol=None, class_weight='balanced',
                                 verbose=3)
        elif model_type == self.MLP_CLASSIFER:
            return MLPClassifier(solver='lbfgs', alpha=1e-5,
                                 verbose=True)
        elif model_type == self. SGD_REGRESSOR:
            return SGDRegressor(loss='squared_loss', penalty='l1',
                                alpha=1e-3, random_state=42,
                                max_iter=5, tol=None,
                                verbose=3)

    def train(self, training):
        self.text_clf.fit(training.inputs, training.targets)

    def test(self, test):
        inputs = list(test.inputs)

        predicted = self.text_clf.predict(inputs)

        df = pd.DataFrame(data={'actual': test.targets, 'predicted': predicted})
        for k, v in df.where(df.actual == df.predicted).groupby([df.actual, df.predicted]).groups.items():
            print(k, inputs[v[0]])

        print()
        print(metrics.classification_report(test.targets, predicted))
        print(metrics.confusion_matrix(test.targets, predicted))

    def visualize(self):
        print()

        v = list(self.text_clf.named_steps['vect'].vocabulary_.keys())

        if self.model_type == self.SGD_CLASSIFER:
            for target, coeffs in enumerate(self.text_clf.named_steps['clf'].coef_):
                top = np.argsort(coeffs)[-10:]
                print(u"{}\n{}".format(target, u"\n".join([u"{} {}".format(v[i], coeffs[i]) for i in top])))
        elif self.model_type == self.MLP_CLASSIFER:
            for target, out_layer in enumerate(self.text_clf.named_steps['clf'].coefs_[1].T):
                max_node = np.argmax(out_layer)
                in_max = self.text_clf.named_steps['clf'].coefs_[0].T[max_node]
                top = np.argsort(in_max)[-5:]
                print(u"{} max {}\n{}".format(self.text_clf.named_steps['clf'].classes_[target],
                                              out_layer[max_node],
                                              u"\n".join([u"{} {}".format(v[i], in_max[i]) for i in top])))
                min_node = np.argmin(out_layer)
                in_min = self.text_clf.named_steps['clf'].coefs_[0].T[min_node]
                top = np.argsort(in_min)[-5:]
                print(u"{} min {}\n{}".format(self.text_clf.named_steps['clf'].classes_[target],
                                              out_layer[min_node],
                                              u"\n".join([u"{} {}".format(v[i], in_max[i]) for i in top])))
        print()
