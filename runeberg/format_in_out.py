#from keras.preprocessing.text import text_to_word_sequence, one_hot
import os

class Format:

    def __init__(self, filename):
        self.filename=filename

    def convert_to_indices(self, z, unique_words):
        new_z=[]
        for z_i in z:
            new_z.append([unique_words[w] for w in z_i])
        
        return new_z


    def get_input_output(self):
        folder_path=self.filename
        files=os.listdir(folder_path)
        x=[]
        y=[]
        labels=set()
        unique_words=set()
        for filename in files:
            path='{}/{}'.format(folder_path,filename)
            with open(path) as f:
                text=f.read()

            lines=text.split('\n')
            year=lines[0]
            labels.add(year)

            sentences=lines[1:]
            # Split all sentences into lists of words, if-statement is to remove empty strings
            split_sentences=[[w for w in s.split(' ') if w!=''] for s in sentences]
            unique_words.update([w for s in split_sentences for w in s])
            #print(year, len(sentences))
            x=x+split_sentences
            y=y+[[year] for i in range(len(split_sentences))]
            #print(len(x), len(y))

        labels = sorted(list(set(labels)))
        return (x, y, unique_words, labels)

    def get_formated_data(self, offset):
        x, y, unique_words, labels = self.get_input_output()
        print(labels)
        #print(unique_words[1:100])
        #print(x[0:4])
        word_to_ind={}
        ind_to_word={}
        for index, word in enumerate(unique_words):
            i=index+offset
            word_to_ind[word]=i
            ind_to_word[i]=word

        #unique_words=dict.fromkeys(unique_words, 0)
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
