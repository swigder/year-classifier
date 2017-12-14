#from keras.preprocessing.text import text_to_word_sequence, one_hot
import os
from os.path import join, isfile
from sklearn.feature_extraction.text import CountVectorizer
import re

class Format:

    def __init__(self, filename):
        self.filename=filename
        self.max_sentences=8000

    def convert_to_indices(self, z, unique_words):
        new_z=[]
        for z_i in z:
            new_z.append([unique_words[w] for w in z_i if w in unique_words])
        
        return new_z


    def get_input_output(self):
        folder_path=self.filename
        files=os.listdir(folder_path)
        files=[fn for fn in files if isfile(join(folder_path,fn))]
        data=[]
        labels=set()
        sentences=[]
        for filename in files:
            path='{}/{}'.format(folder_path,filename)
            with open(path) as f:
                year=f.readline()[:-1]
                text=f.read()

            s=text.split('\n')[:self.max_sentences]
            sentences+=s
            data.append((year, s))
            labels.add(year)


        print("len {}".format(len(sentences)))
        x=[]
        y=[]

        count_vec=CountVectorizer(max_df=.95, min_df=0.0001, token_pattern=r"(?u)\b[A-ZÅÄÖa-zåäö][A-ZÅÄÖa-zåäö]+\b")
        count_vec.fit(sentences)
        vocab=count_vec.vocabulary_
        tokenizer=count_vec.build_tokenizer()
        for d in data:
            # Split all sentences into lists of words, if-statement is to remove empty strings
            split_sentences=[list(filter(lambda x: len(x)>0, [w for w in tokenizer(s) if w in vocab and w!='' and w!=' '])) for s in d[1]]
            #print(year, len(sentences))
            x=x+split_sentences
            y=y+[[d[0]] for i in range(len(split_sentences))]

        labels = sorted(list(set(labels)))
        return (x, y, vocab, labels)

    def get_formated_data(self, offset, word_to_ind=None, label_tr=None):
        x, y, unique_words, labels = self.get_input_output()
        print(labels)
        #print(unique_words[1:100])
        #print(x[0:4])
        if word_to_ind==None:
            word_to_ind={}
            ind_to_word={}
            for index, word in enumerate(unique_words.keys()):
                i=index+offset
                word_to_ind[word]=i
                ind_to_word[i]=word

        else:
            ind_to_word=None

        #unique_words=dict.fromkeys(unique_words, 0)
        if label_tr!=None:
            labels=label_tr

        new_x=self.convert_to_indices(x, word_to_ind)
        label_dict={k:i for i, k in enumerate(labels)}
        new_y=self.convert_to_indices(y, label_dict)
        #print(x[0:5])
        return new_x, new_y, word_to_ind, ind_to_word, labels

    def keras_enc(self):
        x, y, unique_words, labels = self.get_input_output()



def main():
    x, y, word_to_ind, ind_to_word, labels=Format().get_formated_data()
    print(len(word_to_ind))
    print([ind_to_word[i] for x_i in x[0:5] for i in x_i])
    print([labels[i] for y_i in y[0:5] for i in y_i])

if __name__=='__main__':
    main()
