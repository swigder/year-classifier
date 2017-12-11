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
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.set_printoptions(threshold=np.nan)

from format_in_out import Format

NUM_EPOCHS=10
BATCH_SIZE=64
LEARNING_RATE=0.001
TRAIN_SIZE=10000
VAL_SIZE=8000
TEST_SIZE=8000


# Load data
#x, y, word_to_ind, ind_to_word, labels=Format('/tmp/dataset-1/training').get_formated_data(0)
#x_test, y_test, word_to_ind_test, ind_to_word_test, labels_test=Format('/tmp/dataset-1/test').get_formated_data(0)
data_folder_name='/tmp/dataset-p2-s10000-min100-max1000'
x, y, word_to_ind, ind_to_word, labels=Format(data_folder_name+'/training').get_formated_data(0)
x_test, y_test, word_to_ind_test, ind_to_word_test, labels_test=Format(data_folder_name+'/test').get_formated_data(0)
TEST_SIZE=len(x_test)
print("vocab = {}".format(len(word_to_ind)))
#x, y, word_to_ind, ind_to_word, labels=Format('data/formated').get_formated_data(0)

"""
max_len=100
new_x=[]
new_y=[]
for i, x_i in enumerate(x):
    #print(len(x_i))
    if len(x_i)>max_len:
        nl=[x_i[i:i+max_len] for i in range(0, len(x_i), max_len)]
        new_x+=nl
        new_y+=[y[i]]*len(nl)
    else:
        new_x.append(x_i)
        new_y.append(y[i])

print(len(new_x))
print(len(new_y))
print("max len = "+str(max_len))

x=new_x
y=new_y
"""

# Compute document frequency
"""
df={i: 0 for i in ind_to_word.keys()}
se=[]
for x_i in x:
    se.append(set(x_i))

for i, ind in enumerate(ind_to_word.keys()):
    for j in se:
        if ind in j:
            df[ind]+=1
            """

"""
#calculate words frequencies per document
word_frequencies = [Counter(x_i) for x_i in x]
#print(word_frequencies)

#calculate document frequency
df= {i: 0 for i in ind_to_word}
for word_frequency in word_frequencies:
    for wf in word_frequency.keys():
        df[wf]+=1
    
        
print("before {}".format(len(df)))
df={f:0 for f in df.keys() if df[f]>5}
print("after {}".format(len(df)))
new_x=[]
for x_i in x:
    new_x.append([w for w in x_i if w in df])
x=new_x
"""

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

x_val=x[0:VAL_SIZE]
y_val=y[0:VAL_SIZE]
#print(x_test)
#print(y_test)
x=x[VAL_SIZE:]
y=y[VAL_SIZE:]
print(x.shape)
timesteps=x.shape[1]
features=100

lmo=load_model('emb_model.h5')

#model = Sequential()
inputs=Input(shape=(timesteps, ))
#model.add(Embedding(len(word_to_ind), features, input_length=timesteps))
layer=Embedding(len(word_to_ind), features, input_length=timesteps, name='emb_layer')(inputs)
#layer=Embedding(len(word_to_ind), features, input_length=timesteps, name='emb_layer', trainable=False, weights=lmo.get_layer('emb_layer').get_weights())(inputs)

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
"""
#layer2=AveragePooling1D()(layer2)
layer2=Conv1D(128, 1, activation='relu')(layer2)
layer2=MaxPooling1D()(layer2)
#layer2=AveragePooling1D()(layer2)
layer2=Conv1D(128, 3, activation='relu')(layer2)
layer2=MaxPooling1D()(layer2)
"""

layer=layer2 #Concatenate()([layer1, layer2])
#model.add(Conv1D(64, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(MaxPooling1D())
#model.add(AveragePooling1D())
#model.add(Conv1D(16, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(AveragePooling1D())
#model.add(Conv1D(8, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(AveragePooling1D())
layer=Flatten()(layer)
#layer=BatchNormalization()(layer)
layer=Dropout(0.5)(layer)
predictions=Dense(len(labels), activation='softmax',kernel_regularizer=regularizers.l2(0.01))(layer)
model=Model(inputs=inputs, outputs=predictions)
#input_array = np.random.randint(1000, size=(32, 10))

opt=Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

print("time: {}, epochs: {}, learning rate: {}, training size: {}, test size: {}".format(datetime.datetime.utcnow(), NUM_EPOCHS, LEARNING_RATE,TRAIN_SIZE,TEST_SIZE))
print(model.summary())

model.fit(x, y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
"""
k=0
for i in range(3):
    print("Epoch = {}".format(i))
    for x_i, y_i in zip(x,y):
        model.train_on_batch(np.array([x_i]), np.reshape(np.array(y_i), (1, len(labels))))
        if k%1000==0:
            score,acc=model.evaluate(x_val,  y_val, batch_size=1)
            print('score = {}, accuracy = {}'.format(score, acc))

        k+=1
        print(k)
        """

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
print(metrics.classification_report(test.targets, predicted))


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
