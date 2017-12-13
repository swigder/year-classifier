from keras.layers.core import Masking
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Dense, Activation, LSTM, BatchNormalization, Bidirectional
from keras.models import load_model, Model

class LstmModel:

    def get(features, vocab_size, timesteps, output_size):
        model = Sequential()
        model.add(Embedding(vocab_size+1, features, mask_zero=True))
        # the model will take as input an integer matrix of size (batch, input_length).
        # the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
        # now model.output_shape == (None, 10, 64), where None is the batch dimension.

        model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
        model.add(Bidirectional(LSTM(10, activation='relu')))
        model.add(BatchNormalization())
        model.add(Dense(output_size, activation='softmax'))
        return model
