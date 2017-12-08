from keras.layers.core import Masking
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import regularizers
from keras.layers import Embedding, Dense, Activation, LSTM, BatchNormalization, Bidirectional, Conv1D, Flatten, MaxPooling1D, AveragePooling1D, Dropout, Input, Concatenate
from keras.models import load_model, Model

import numpy as np
import os
from sklearn.metrics import confusion_matrix
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.set_printoptions(threshold=np.nan)

from format_in_out import Format

NUM_EPOCHS=10
BATCH_SIZE=32
LEARNING_RATE=0.001
TRAIN_SIZE=10000
TEST_SIZE=2000


# Load data
#x, y, word_to_ind, ind_to_word, labels=Format('/tmp/formated_data_small').get_formated_data(0)
x, y, word_to_ind, ind_to_word, labels=Format('data/formated').get_formated_data(0)

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

x_test=x[0:TEST_SIZE]
y_test=y[0:TEST_SIZE]
#print(x_test)
#print(y_test)
x=x[TEST_SIZE:TEST_SIZE+TRAIN_SIZE]
y=y[TEST_SIZE:TEST_SIZE+TRAIN_SIZE]
#print(x.shape)
timesteps=x.shape[1]
features=300

lmo=load_model('emb_model.h5')

#model = Sequential()
inputs=Input(shape=(timesteps, ))
#model.add(Embedding(len(word_to_ind), features, input_length=timesteps))
layer=Embedding(len(word_to_ind), features, input_length=timesteps, name='emb_layer', trainable=False, weights=lmo.get_layer('emb_layer').get_weights())(inputs)

# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

#model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
#layer1=Conv1D(64, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(layer)
#layer1=Conv1D(64, 1, activation='relu',kernel_regularizer=regularizers.l2(0.01))(layer1)
#layer1=Conv1D(64, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(layer1)
#layer1=AveragePooling1D()(layer1)

layer2=Conv1D(128, 3, activation='relu')(layer)
layer2=MaxPooling1D()(layer2)
#layer2=AveragePooling1D()(layer2)
layer2=Conv1D(128, 1, activation='relu')(layer2)
layer2=MaxPooling1D()(layer2)
#layer2=AveragePooling1D()(layer2)
layer2=Conv1D(128, 3, activation='relu')(layer2)
layer2=MaxPooling1D()(layer2)

layer=layer2 #Concatenate()([layer1, layer2])
#model.add(Conv1D(64, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(MaxPooling1D())
#model.add(AveragePooling1D())
#model.add(Conv1D(16, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(AveragePooling1D())
#model.add(Conv1D(8, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(AveragePooling1D())
layer=Flatten()(layer)
layer=BatchNormalization()(layer)
layer=Dropout(0.5)(layer)
predictions=Dense(14, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(layer)
model=Model(inputs=inputs, outputs=predictions)
#input_array = np.random.randint(1000, size=(32, 10))

opt=Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

print("time: {}, epochs: {}, learning rate: {}, training size: {}, test size: {}".format(datetime.datetime.utcnow(), NUM_EPOCHS, LEARNING_RATE,TRAIN_SIZE,TEST_SIZE))
print(model.summary())

model.fit(x, y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

out=model.predict(x_test)
#print(out[0:100])
#print(y_test[0:100])
score,acc=model.evaluate(x_test,  y_test, batch_size=32)
print(score, acc)
lab=np.argmax(y_test, axis=1)
pred_lab=np.argmax(out, axis=1)
print(lab[0:100])
print(pred_lab[0:100])
conf_matrix=confusion_matrix(lab, pred_lab)
print(conf_matrix)

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
