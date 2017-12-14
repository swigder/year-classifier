from keras.layers.core import Masking
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Dense, Activation, LSTM, BatchNormalization, Bidirectional
from keras.models import load_model, Model

class LstmModel:

    def get(features, vocab_size, timesteps, output_size):
        model = Sequential()
        model.add(Embedding(vocab_size+1, features, mask_zero=True))

        model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
        model.add(Bidirectional(LSTM(10, activation='relu')))
        model.add(BatchNormalization())
        model.add(Dense(output_size, activation='softmax'))
        return model
