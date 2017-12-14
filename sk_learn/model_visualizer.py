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
        # todo warning! this is not supported
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

        # calculate
        results = pd.DataFrame(columns=self.classes, index=self.vocabulary, data=self.coeffs.T)
        var = results.var(axis=1)
        diff_top_two = results.max(axis=1) - results.quantile(11/12, axis=1)
        top_years = results.loc[diff_top_two.sort_values(ascending=False)._index].idxmax(axis=1)

        # outlier words
        outliers_values = diff_top_two.nlargest(20)
        outlier_years = results.loc[outliers_values._index].idxmax(axis=1)
        plt.scatter(outlier_years, outliers_values)
        for word in outliers_values._index:
            plt.annotate(word, (outlier_years[word], outliers_values[word]))
        for year in self.classes:
            if year not in outlier_years.values:
                top_outlier = top_years.where(top_years == year).idxmax()
                plt.scatter(year, diff_top_two[top_outlier], c='g')
                plt.annotate(top_outlier, (year, diff_top_two[top_outlier]))
        plt.xlim((1750, 2050))
        plt.xlabel('Year with highest likelihood')
        plt.ylabel('Difference in log-likelihood of top two years')

        # heatmap
        plt.figure()
        sns.heatmap(results.loc[var.nlargest(50)._index])
        plt.yticks(rotation=0)
        plt.show()

    def outlying(self, row, iq_range=.5):
        pcnt = (1 - iq_range) / 2
        qlow, median, qhigh = row.quantile([pcnt, 0.50, 1 - pcnt])
        iqr = qhigh - qlow
        return row.max() - iqr
