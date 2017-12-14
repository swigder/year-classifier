import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ModelVisualizer:
    def __init__(self, model):
        self.classes = model.text_clf.named_steps['clf'].classes_
        self.coeffs = model.text_clf.named_steps['clf'].coef_
        vocab = model.text_clf.named_steps['vect'].vocabulary_
        terms = np.array(list(vocab.keys()))
        indices = np.array(list(vocab.values()))
        self.vocabulary = terms[np.argsort(indices)]

    def visualize_sgd_classifier(self):
        # todo warning! this is not fully supported
        for target, coeffs in enumerate(self.coeffs):
            top = np.argsort(coeffs)[-10:]
            print(u"{}\n{}".format(self.classes[target], u"\n".join([u"{} {}".format(v[i], coeffs[i]) for i in top])))

    def visualize_mlp_classifier(self):
        # todo warning! this is not fully supported
        layer0 = self.coeffs[0]
        layer1 = self.coeffs[1]
        vocab_var = np.var(layer0, axis=1)
        n_words_to_examine = 50
        ind = np.argpartition(vocab_var, -n_words_to_examine)[-n_words_to_examine:]
        ind = ind[np.argsort(vocab_var[ind])][::-1]
        for i in ind:
            top_hidden = np.argmax(layer0[i])
            hidden_target = np.argmax(layer1[top_hidden])
            print(self.vocabulary[i], np.var(layer0[i]), self.classes[hidden_target], self.text_clf.predict([v[i]]))

    def visualize_naive_bayes(self):
        sns.set()
        vocab_var = np.var(self.coeffs, axis=0)
        n_words_to_examine = 50
        ind = np.argpartition(vocab_var, -n_words_to_examine)[-n_words_to_examine:]
        ind = ind[np.argsort(vocab_var[ind])][::-1]
        results = pd.DataFrame(columns=self.classes)
        for i in ind:
            word_coeffs = self.coeffs[:, i]
            year, variance = self.classes[np.argmax(word_coeffs)], np.var(word_coeffs)
            plt.scatter(year, variance)
            plt.annotate(self.vocabulary[i], (year, variance))
            print(self.vocabulary[i], year, variance, word_coeffs)
            results.loc[self.vocabulary[i]] = word_coeffs
        plt.xlim((1750, 2020))
        plt.xlabel('Year with highest likelihood')
        plt.ylabel('Variance of log-likelihood')
        plt.figure()
        sns.heatmap(results)
        plt.yticks(rotation=0)
        plt.show()
