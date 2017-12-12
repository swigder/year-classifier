from keras.layers.core import Masking
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Dense, Activation, LSTM, BatchNormalization, Bidirectional
from keras.models import load_model, Model
from keras.callbacks import TensorBoard

import numpy as np
import time
import os
from sklearn.metrics import confusion_matrix
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.set_printoptions(threshold=np.nan)

from format_in_out import Format

NUM_EPOCHS=10
TRAIN_SIZE=30000
TEST_SIZE=5000
LEARNING_RATE=0.0001
BATCH_SIZE=64

# Load data
data_folder_name='/tmp/dataset-p2-s30000-min100-max2000' #'/tmp/dataset-p2-s10000-min100-max1000'
x, y, word_to_ind, ind_to_word, labels=Format(data_folder_name+'/training').get_formated_data(0)
x_test, y_test, word_to_ind_test, ind_to_word_test, labels_test=Format(data_folder_name+'/test').get_formated_data(0)
TEST_SIZE=len(x_test)
print("vocab = {}".format(len(word_to_ind)))

y_new=np.zeros((len(y), len(labels)))

for i, y_i in enumerate(y):
    y_new[i, y_i[0]]=1

#print(y_new[0:10])
y=y_new
x=sequence.pad_sequences(x)
x=np.array(x)

perm=np.random.permutation(x.shape[0])
x=x[perm]
y=y[perm]

print(x.shape)
timesteps=x.shape[1]
features=100

lmo=load_model('emb_model.h5')

model = Sequential()
model.add(Embedding(len(word_to_ind)+1, features, mask_zero=True))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(20, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(len(labels), activation='softmax'))

#input_array = np.random.randint(1000, size=(32, 10))

opt=Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
print("time: {}, epochs: {}, learning rate: {}, training size: {}, test size: {}, batch size={}".format(datetime.datetime.utcnow(), NUM_EPOCHS, LEARNING_RATE,TRAIN_SIZE,TEST_SIZE, BATCH_SIZE))
print(model.summary())

log_directory='/tmp/logs_lstm_'+time.strftime("%H%M")
tb=TensorBoard(log_dir=log_directory, embeddings_freq=1, embeddings_metadata={'emb_layer': '/tmp/embedding_metadata'})

model.fit(x, y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[tb])

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


"""
tot=np.sum(np.sum(confusion))
accuracy=sum([confusion[i][i] for i in range(self.FEATURES)])/tot
TP=confusion[1][1]
TN=confusion[0][0]
FP=confusion[1][0]
FN=confusion[0][1]
recall=TP/(TP+FN)
precision=TP/(TP+FP)
print("accuracy = {}, precision = {}, recall = {}".format(accuracy, precision, recall))
"""
#output_array = model.predict(input_array)
#print(output_array.shape)
