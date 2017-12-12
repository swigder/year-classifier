from keras.layers.core import Masking
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import regularizers
from keras.layers import Embedding, Dense, Activation, LSTM, BatchNormalization, Bidirectional, Conv1D, Flatten, MaxPooling1D, AveragePooling1D, Dropout, Input, Concatenate
from keras.models import load_model, Model
from keras.callbacks import TensorBoard

import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import datetime
import time
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.set_printoptions(threshold=np.nan)

from format_in_out import Format

NUM_EPOCHS=6
BATCH_SIZE=64
LEARNING_RATE=0.0001
TRAIN_SIZE=4000
VAL_SIZE=8000
TEST_SIZE=8000


# Load data
#x, y, word_to_ind, ind_to_word, labels=Format('/tmp/dataset-1/training').get_formated_data(0)
#x_test, y_test, word_to_ind_test, ind_to_word_test, labels_test=Format('/tmp/dataset-1/test').get_formated_data(0)
data_folder_name='/tmp/dataset-p2-s30000-min100-max2000' #'/tmp/dataset-p2-s10000-min100-max1000'
x, y, word_to_ind, ind_to_word, labels=Format(data_folder_name+'/training').get_formated_data(0)
x_test, y_test, word_to_ind_test, ind_to_word_test, labels_test=Format(data_folder_name+'/test').get_formated_data(0)

# Remove all words not in training set
x_test=[[w for w in s if w in ind_to_word] for s in x_test]


TEST_SIZE=len(x_test)
print("vocab = {}".format(len(word_to_ind)))

with open('/tmp/embedding_metadata', 'w+t') as f:
    f.write('Word\tIndex\n')
    for i, w in enumerate(word_to_ind.keys()):
        f.write('{}\t{}\n'.format(w, word_to_ind[w]))

def to_onehot(z):
    z_new=np.zeros((len(z), len(labels)))

    for i, z_i in enumerate(z):
        z_new[i, z_i[0]]=1

    return z_new

#print(y_new[0:10])
y=to_onehot(y)
y_test=to_onehot(y_test)

x_tot=x+x_test
x_tot=sequence.pad_sequences(x_tot)
x_tot=np.array(x_tot)

x=x_tot[0:len(x)]

x_test=x_tot[len(x):]
x_test=np.array(x_test)

perm=np.random.permutation(x.shape[0])
x=x[perm]
y=y[perm]

"""
x_test=x_test[0:TEST_SIZE]
y_test=y_test[0:TEST_SIZE]
x=x[TEST_SIZE:TEST_SIZE+TRAIN_SIZE]
y=y[TEST_SIZE:TEST_SIZE+TRAIN_SIZE]
"""

perm=np.random.permutation(x_test.shape[0])
#x_test=x_test[perm]
#y_test=y_test[perm]
#x=x[0:5000]
#y=y[0:5000]

print(x.shape)
timesteps=x.shape[1]
features=50

try:
    lmo=load_model('/tmp/emb_model.h5')
except:
    print('embedding model not found')


model_name=sys.argv[1]
from conv_one_layer_model import ConvModel
from multi_conv_model import MultiConvModel
from lstm_model import LstmModel
models={'1conv':, 'multiconv':, 'lstm':}

model=models[model_name].get()

opt=Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

print("time: {}, epochs: {}, learning rate: {}, training size: {}, test size: {}".format(datetime.datetime.utcnow(), NUM_EPOCHS, LEARNING_RATE,TRAIN_SIZE,TEST_SIZE))
print(model.summary())

log_directory='/tmp/logs_conv_'+time.strftime("%H%M")
tb=TensorBoard(log_dir=log_directory, embeddings_freq=1, embeddings_metadata={'emb_layer': '/tmp/embedding_metadata'})

model.fit(x, y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[tb])

#print(out[0:100])
#print(y_test[0:100])
out=model.predict(x_test)
score,acc=model.evaluate(x_test,  y_test, batch_size=BATCH_SIZE)
print(score, acc)
lab=np.argmax(y_test, axis=1)
pred_lab=np.argmax(out, axis=1)
print(lab[0:100])
print(pred_lab[0:100])
conf_matrix=confusion_matrix(lab, pred_lab)
print(conf_matrix)
print(metrics.classification_report(lab, pred_lab))


