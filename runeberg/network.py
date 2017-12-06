from keras.models import Sequential
from keras.layers import Embedding, Dense, Activation
import numpy as np

from format_in_out import Format

# Load data
x, y, word_to_ind, ind_to_word, labels=Format().get_formated_data()

model = Sequential()
model.add(Embedding(len(word_to_ind), 64, input_length=10))
#model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

#input_array = np.random.randint(1000, size=(32, 10))
input_array = x

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
