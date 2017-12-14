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
        inputs=Input(shape=(timesteps, ))
        layer=Embedding(vocab_size, features, input_length=timesteps, name='emb_layer')(inputs)

        layer1=Conv1D(32, 3, padding='same', activation='relu')(layer)
        layer=MaxPooling1D()(layer1)
        layer=Flatten()(layer)
        predictions=Dense(output_size, activation='softmax')(layer)
        model=Model(inputs=inputs, outputs=predictions)
        return model
