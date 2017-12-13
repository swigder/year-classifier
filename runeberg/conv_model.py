from keras.layers.core import Masking
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import regularizers
from keras.layers import Embedding, Dense, Activation, LSTM, BatchNormalization, Bidirectional, Conv1D, Flatten, MaxPooling1D, AveragePooling1D, Dropout, Input, Concatenate
from keras.models import load_model, Model
from keras.callbacks import TensorBoard

class ConvModel:
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
        layer1=Conv1D(128, 5, padding='same', activation='relu')(layer)
        layer=MaxPooling1D()(layer1)
        layer=Flatten()(layer)
        #layer=Dropout(0.5)(layer)
        predictions=Dense(output_size, activation='softmax')(layer)
        #predictions=Dense(output_size, activation='softmax')(layer)
        model=Model(inputs=inputs, outputs=predictions)
        #input_array = np.random.randint(1000, size=(32, 10))
        return model
