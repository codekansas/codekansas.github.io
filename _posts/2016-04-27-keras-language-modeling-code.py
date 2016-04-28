import numpy as np

rng = np.random.RandomState(42)

# parameters
input_dims, output_dims = 10, 1
sequence_length = 20
n_test = 10

# generate some random data to train on
get_rand = lambda *shape: np.asarray(rng.rand(*shape) > 0.5, dtype='float32')
X_data = np.asarray([get_rand(sequence_length, input_dims) for _ in range(n_test)])
y_data = np.asarray([get_rand(output_dims,) for _ in range(n_test)])

# put together rnn models
from keras.layers import Input
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.models import Model
import theano

input_sequence = Input(shape=(sequence_length, input_dims,), dtype='float32')

vanilla = SimpleRNN(output_dims, return_sequences=False)(input_sequence)
lstm = LSTM(output_dims, return_sequences=False)(input_sequence)
gru = GRU(output_dims, return_sequences=False)(input_sequence)
rnns = [vanilla, lstm, gru]

# train the models
for rnn in rnns:
    model = Model(input=[input_sequence], output=rnn)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    model.fit([X_data], [y_data], nb_epoch=1000)

