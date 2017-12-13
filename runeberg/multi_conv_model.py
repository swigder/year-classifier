from keras.layers.core import Masking
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import regularizers
from keras.layers import Embedding, Dense, Activation, LSTM, BatchNormalization, Bidirectional, Conv1D, Flatten, MaxPooling1D, AveragePooling1D, Dropout, Input, Concatenate
from keras.models import load_model, Model
from keras.callbacks import TensorBoard


class MultiConvModel:
    def get(features, vocab_size, timesteps, output_size):
        #model = Sequential()
        inputs=Input(shape=(timesteps, ))
        #model.add(Embedding(len(word_to_ind), features, input_length=timesteps))
        layer=Embedding(vocab_size, features, input_length=timesteps, name='emb_layer')(inputs)
        #layer=Embedding(len(word_to_ind), features, input_length=timesteps, name='emb_layer', trainable=False, weights=lmo.get_layer('emb_layer').get_weights())(inputs)

        # the model will take as input an integer matrix of size (batch, input_length).
        # the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
        # now model.output_shape == (None, 10, 64), where None is the batch dimension.

        #model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
        conv1=Conv1D(32, 3, padding='same', activation='relu')(layer)
        conv2=Conv1D(32, 5, padding='same', activation='relu')(layer)
        conv3=Conv1D(32, 7, padding='same', activation='relu')(layer)
        conv4=MaxPooling1D(pool_size=3, strides=1, padding='same')(layer)

        layer=Concatenate()([conv1, conv2, conv3, conv4])
        layer=Flatten()(layer)
        #layer=BatchNormalization()(layer)
        #layer=Dropout(0.5)(layer)
        predictions=Dense(output_size, activation='softmax')(layer)
        model=Model(inputs=inputs, outputs=predictions)
        #input_array = np.random.randint(1000, size=(32, 10))
        return model

