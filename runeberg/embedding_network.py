from keras.layers.core import Masking
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import regularizers
from keras.layers import Embedding, Dense, Activation, LSTM, BatchNormalization, Bidirectional, Conv1D, Flatten, MaxPooling1D, AveragePooling1D, Dropout
import numpy as np
from keras.models import load_model
import os
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.set_printoptions(threshold=np.nan)

from format_in_out import Format

NUM_EPOCHS=10
BATCH_SIZE=64
LEARNING_RATE=0.001
TRAIN_SIZE=36000
TEST_SIZE=200


# Load data
#x, y, word_to_ind, ind_to_word, labels=Format('/tmp/dataset-1/training').get_formated_data(0)
#x_test, y_test, word_to_ind_test, ind_to_word_test, labels_test=Format('/tmp/dataset-1/test').get_formated_data(0)
data_folder_name='/tmp/dataset-p2-s10000-min100-max1000'
x, y, word_to_ind, ind_to_word, labels=Format(data_folder_name+'/training').get_formated_data(0)
x_test, y_test, word_to_ind_test, ind_to_word_test, labels_test=Format(data_folder_name+'/test').get_formated_data(0)

y_new=np.zeros((len(y), len(labels)))

for i, y_i in enumerate(y):
    y_new[i, y_i[0]]=1

y=y_new

#print(y_new[0:10])
#y=y_new
x=sequence.pad_sequences(x)
x=np.array(x)
print(x.shape)

perm=np.random.permutation(x.shape[0])
x=x[perm]
y=y[perm]

timesteps=x.shape[1]
features=100

#lmo=load_model('emb_model.h5')

model = Sequential()
#model.add(Embedding(len(word_to_ind), features, input_length=timesteps, name='emb_layer', trainable=False, weights=lmo.get_layer('emb_layer').get_weights()))
model.add(Embedding(len(word_to_ind), features, input_shape=(timesteps,), name='emb_layer'))
#model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

#model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(len(labels), activation='softmax'))

#input_array = np.random.randint(1000, size=(32, 10))

opt=Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

print("time: {}, epochs: {}, learning rate: {}, training size: {}, test size: {}".format(datetime.datetime.utcnow(), NUM_EPOCHS, LEARNING_RATE,TRAIN_SIZE,TEST_SIZE))
print(model.summary())

"""
for i in range(3):
    print("Epoch = {}".format(i))
    for x_i, y_i in zip(x,y):
        onehot_encoded = to_categorical(x_i, num_classes=len(word_to_ind))
        print(x_i, onehot_encoded)
        model.fit(np.array([onehot_encoded]), [y_i], epochs=1, batch_size=1)
        """

model.fit(x, y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)
model.save('/tmp/emb_model.h5')
#em_weights=model.get_layer('emb_layer').get_weights()
#print(em_weights)

#print(out[0:100])
#print(y_test[0:100])
score,acc=model.evaluate(x_test,  y_test, batch_size=64)
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
