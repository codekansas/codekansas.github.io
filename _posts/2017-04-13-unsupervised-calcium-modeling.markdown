---
layout: post
title: "Deep Learning for Calcium Imaging"
date: 2017-04-13 12:00:00
categories: machine-learning
keywords:
 - Calcium Imaging
 - Deep Learning
 - Autoencoder
 - Keras
excerpt: >
  A tutorial on applying supervised and unsupervised deep learning methods to calcium imaging. These methods are applicable to other time-series modeling problems.
image: /resources/index/calcium_autoencoder.png
links:
 - View Code: https://github.com/codekansas/calcium-gan
 - Spikefinder Competition: http://spikefinder.codeneuro.org/
---

# Abstract

A key idea in neuroscience is that neurons encode senses and behavior with "spikes" of electrical activity. These spikes initiate in the cell body and propagate down the wire-like axon to "synapses", or connections with other neurons. These neurons integrate the signal with many other incoming spikes from many other neurons. For nearly a century, the most common way to record spikes has been with extracellular electrodes. This type of electrode is basically a wire that in close proximity to a neuron. It can record the electrical potential across the membrane - like a tiny microphone.

While single-unit recordings of extracellular activity have given us important insights into the neural code, they do not paint a complete picture. Importantly they do not allow us to easily see what might be encoded at a population level. Multielectrode arrays can help with this, but even with arrays we may miss important information about the population, like which types of neurons in a region connect with each other and which do not spike at all during a given behavior.

One alternative to recording neurons with electrodes is imaging calcium concentration in the cell. It is known that calcium levels increase when a neuron spikes. Calcium indicator dyes bind to calcium and upon doing so, the indicator molecule adopts a conformation that is fluorescent.

{% include image.html description="Graphic visualizing calcium imaging in a population of neurons. When a neuron spikes, the calcium concentration increases, causing the indicator dye to change color. For this tutorial, the data traces represent one neuron each. The calcium trace is the average fluorescence of the neuron, and the spike trace is the membrane potential of the neuron. The spike inference task consists of predicting the membrane potential, and therefore the spiking behavior, from the calcium fluorescence." url="/resources/calcium_autoencoder/movie.gif" %}

"Spike inference" from calcium fluorescence is a non-trivial task. The relationship between the amount of calcium in the cell and the cell's membrane potential is non-linear and time-dependent; calcium levels do not necessarily track membrane voltage closely. Binding of the indicator dyes to calcium is itself a reaction that takes time to occur. In addition, there are often sources of noise in the recordings and the results can depend on the experimenter's choice of the region of a neuron to image (although most protocols attempt to minimize noise due to "region of interest").

In this tutorial, we will use recent methods from deep learning to characterize the distribution of fluorescence signals that correspond to spiking and to non-spiking periods. In this process, we will build up our understand of deep learning methods which can be applied to other tasks.

## Preparation

This tutorial assumes you have an idea of the fundamentals of deep learning. We will use the deep learning framework [Keras](https://keras.io/) to build our models. Take a look through the documentation to understand how things work.

The tutorial will use the training data provided from the [SpikeFinder competition](http://spikefinder.codeneuro.org/). This data is hosted on AWS [here](https://s3.amazonaws.com/neuro.datasets/challenges/spikefinder/spikefinder.train.zip) and can be loaded and managed using Pandas.

# Modeling Calcium Fluorescences

First, let's visualize the data we're actually trying to model. This tutorial will use pretty standard dependencies, plus a utils file which is available in the project directory [here](https://github.com/codekansas/calcium-gan/blob/master/utils.py).

```python
import os
import sys

from imp import reload
import utils; reload(utils)

import keras
import numpy as np
import matplotlib.pyplot as plt

# Checks that you've got Keras 2.0.0 installed (for compatibility).
assert keras.__version__ == '2.0.2', 'Invalid Keras version.'
```

    Using TensorFlow backend.


Let's use matplotlib to plot samples from the data.


```python
# Plots the first 5 samples.
fig = plt.figure(figsize=(15, 6))
for i, (c, s) in zip(range(1, 4), utils.iterate_files()):
    plt.subplot(3, 1, i)
    utils.plot_sample(c['0'], s['0'])

plt.tight_layout()
plt.show()

del i, c, s
```


{% include image.html description="Samples of the data that we will plot. The calcium trace is shown in green, and the spike trace is shown in black. The amplitude of the calcium trace represents the intensity of the calcium fluorescence; the amplitude of the spike trace represents the number of spikes that were recorded in a particular time bin." url="/resources/calcium_autoencoder/classifier_3_0.png" %}


Now that we've got an idea of what the data looks like, let's parse a dataset. We will partition the calcium fluorescences into two parts: the part where there is at least one spike in an N-bin range, and the part where there isn't. We can then try to predict which calcium traces correspond to a spike and which ones don't.


```python
# Gets five examples of each type.
yes_cal, no_cal, yes_spikes, no_spikes = [], [], [], []
for calcium, spikes, did_spike in utils.partition_data(10, spike_n=1):
    if did_spike:
        yes_cal.append(calcium)
        yes_spikes.append(spikes)
    else:
        no_cal.append(calcium)
        no_spikes.append(spikes)

    if len(yes_spikes) > 5 and len(no_spikes) > 5:
        break

# Plot the data where no spike was observed on the left,
# and the data where a spike was observed on the right.
fig = plt.figure(figsize=(7, 9))
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    utils.plot_sample(no_cal[i], no_spikes[i], t_start=-10, t_end=9, sampling_rate=1)
    plt.title('Sample %d, no associated spike' % i)
    plt.subplot(5, 2, 2 * i + 2);
    utils.plot_sample(yes_cal[i], yes_spikes[i], t_start=-10, t_end=9, sampling_rate=1)
    plt.title('Sample %d, associated spike' % i)

plt.tight_layout()
plt.show()

del yes_cal, no_cal, yes_spikes, no_spikes
```

{% include image.html description="Samples of the data that will be fed to the classifier. We will train a model to take as input the calcium trace (green) and output a 0 if there were no spikes in the middle of the trace, and a 1 if there were spikes. We only care if there were spikes in the middle five time bins (from time -2 to time 2)." url="/resources/calcium_autoencoder/classifier_5_0.png" %}

Great! Now we've got a way of splitting the dataset into "spike observed" and "no spike observed" samples. Let's split all the data up this way, and put them in Numpy arrays so that we can build our model. Since we just care about whether or not a spike was observed at the current timestep, we'll ignore the spike trace. Additionally, let's cache the Numpy arrays somewhere on disk so that we don't have to worry about regenerating the dataset every time we want to use it (since this operation can be kind of slow).


```python
calcium, did_spike = utils.load_dataset()
print('Size of the dataset:')
print('    calcium: %d samples of length %d' % (calcium.shape[0], calcium.shape[1]))
print('    did_spike: %d samples' % did_spike.shape[0])

del calcium, did_spike
```

    Size of the dataset:
        calcium: 514860 samples of length 20
        did_spike: 514860 samples


Great, now all of our data preprocessing is done. Let's get into the specifics of the model. First, we'll build a recurrent neural network classifier and see how well it does. Next, we'll build a [Variational Autoencoder](https://arxiv.org/abs/1606.05908) to build an unsupervised model of the data (in other words, a model that doesn't know if the calcium fluorescence corresponded to a spike).

Supervised learning is Keras' bread and butter. We can very easily build a recurrent neural network classifier to predict when there is and isn't a spike. Our data is nicely structured so that all we have to do is `model.fit([calcium], [did_spike])`. We'll add some more features to the calcium trace to account for non-linearities; these features are the [`DeltaFeature`](https://github.com/codekansas/calcium-gan/blob/master/utils.py#L178) and the [`QuadFeature`](https://github.com/codekansas/calcium-gan/blob/master/utils.py#L196) that are defined in the `utils.py` file. These will help account for non-linear dynamics in the cell.

```python
def build_model(input_len):
    input_calcium = keras.layers.Input(shape=(input_len,), name='input_calcium')

    # This adds some more features that the model can use.
    calcium_rep = keras.layers.Reshape((input_len, 1))(input_calcium)
    calcium_delta = utils.DeltaFeature()(calcium_rep)
    calcium_quad = utils.QuadFeature()(calcium_rep)
    calcium_delta_quad = utils.QuadFeature()(calcium_delta)
    x = keras.layers.Concatenate()([calcium_rep, calcium_delta, calcium_quad])

    # This is the single LSTM layer that performs the classification.
    x = keras.layers.LSTM(64, return_sequences=False)(x)

    output_pred = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputs=[input_calcium], outputs=[output_pred])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def get_evenly_split_dataset(num_samples):
    """Gets an evenly-split sample of the data."""

    calcium, did_spike = utils.load_dataset()

    spike_idxs = np.arange(calcium.shape[0])[did_spike == 1]
    nospike_idxs = np.arange(calcium.shape[0])[did_spike == 0]
    spike_idxs = np.random.choice(spike_idxs, num_samples // 2)
    nospike_idxs = np.random.choice(nospike_idxs, num_samples // 2)
    idxs = np.concatenate([spike_idxs, nospike_idxs])

    return calcium[idxs], did_spike[idxs]

NUM_LSTM_TRAIN = 10000
calcium, did_spike = get_evenly_split_dataset(NUM_LSTM_TRAIN)
model = build_model(calcium.shape[1])

# Trains the model for a single pass.
model.fit([calcium], [did_spike], epochs=5, verbose=2)
print('Done')
```

    Epoch 1/5
    9s - loss: 0.6198 - acc: 0.6800
    Epoch 2/5
    9s - loss: 0.5904 - acc: 0.6943
    Epoch 3/5
    9s - loss: 0.5762 - acc: 0.7093
    Epoch 4/5
    8s - loss: 0.5777 - acc: 0.7110
    Epoch 5/5
    8s - loss: 0.5800 - acc: 0.7070
    Done


We get around 70% accuracy after five epochs with this model. Let's also print a confusion matrix so we can get a bit more information about how the model performed.


```python
preds = model.predict(calcium)
p_s, p_n = preds[did_spike == 1], preds[did_spike == 0]
n_total = calcium.shape[0]

# Computes confusion matrix values.
ss, ns = np.sum(p_s > 0.5) / n_total, np.sum(p_s <= 0.5) / n_total
sn, nn = np.sum(p_n > 0.5) / n_total, np.sum(p_n <= 0.5) / n_total

print('                     spike    no spike')
print('predicted spike    | %.3f  | %.3f' % (ss, ns))
print('predicted no spike | %.3f  | %.3f' % (sn, nn))
```

                         spike    no spike
    predicted spike    | 0.354  | 0.146
    predicted no spike | 0.144  | 0.356


The model does't perform that great; spike inference is clearly a hard problem. Let's take a look at some of the misclassifications.


```python
plt.figure(figsize=(10, 10))

# Gets the false positives and false negatives.
c_s, c_n = calcium[did_spike == 1, :], calcium[did_spike == 0, :]
p_sf, p_nf = np.squeeze(p_s), np.squeeze(p_n)
ns_calc, sn_calc = c_s[p_sf <= 0.5], c_n[p_nf > 0.5]

d = calcium.shape[1] / 2
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    utils.plot_sample(calcium=ns_calc[i],
                      t_start=-d,
                      t_end=d - 1,
                      sampling_rate=1)
    plt.title('Sample %d, false positive' % i)

    plt.subplot(5, 2, 2 * i + 2)
    utils.plot_sample(calcium=sn_calc[i],
                      t_start=-d,
                      t_end=d - 1,
                      sampling_rate=1)
    plt.title('Sample %d, false negative' % i)

plt.tight_layout()
plt.show()
```


{% include image.html description="Visualization of some of the false positives and false negatives from the model. It is hard to distinguish a clear pattern; for a classifier, it seems there is a lot of noise in the data." url="/resources/calcium_autoencoder/classifier_13_0.png" %}


## Autoencoders

Now that we've built a basic classifier for the dataset, let's take a look at an unsupervised deep learning model called an *autoencoder*. An autoencoder is simply a neural network that learns to reconstruct its own inputs. Typically, it learns a mapping to a low-dimensional space, then it learns to take that mapping and reproduce the original data. The simplest example of an autoencoder is Principal Component Analysis; the principal components represent orthogonal manifolds in the data, where the first principal component has the highest variance, the second has the second most variance, and so on. Taking the top-N principal components is a way of doing "dimensionality reduction" on the data. Conceptually, this process captures common features of the dataset. Neural network autoencoders do much the same thing. The diagram below shows the structure of an autoencoder.

{% include image.html description="Diagram of a simple autoencoder, from <a href='https://en.wikipedia.org/wiki/Autoencoder'>Wikipedia</a>. The latent vector Z in the autoencoder is the low-dimensional representation of the data X; for the decoder to perform well, this latent vector must have all the information necessary to represent the data." url="/resources/calcium_autoencoder/autoencoder_structure.png" %}

## Variational Autoencoders

Variational autoencoders are an interesting extension to regular autoencoders. Suppose the latent vector $$Z$$ is a normally-distributed random variable. After training, we could feed our own vector, sampled from a normal distribution, into the decoder neural network, and get a reconstruction back out.

To do this, we just need to make sure the latent vector is normally distributed. We apply a penalty to minimize the KL divergence between $$Z$$ and a normal distribution. KL divergence is simply a measure of how different two distributions are:

$$D_{KL} (P \Vert Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx$$

Given a latent vector with mean $$\mu$$ and variance $$\sigma^2$$, we apply the penalty

$$C_{KL} = -\frac{1 + \sigma - \mu^2 - e^{\sigma}}{2}$$

to push the latent vector towards a normal distribution.

## Building the model

Let's build a very simple autoencoder model using Keras and train it on some sample data. The first thing we need to do is add a new layer to represent the variational part. To do this, we'll reference the variational autoencoder example in the Keras examples directory [here](https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py). We'll also add some extra layers that we can use later.


```python
from __future__ import division
from __future__ import print_function

import os
import sys

from imp import reload
import utils; reload(utils)

from sklearn.decomposition import PCA
from sklearn import svm
import keras
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

# Checks that you've got Keras 2.0.0 installed (for compatibility).
assert keras.__version__ == '2.0.2', 'Invalid Keras version.'


class VariationalLayer(keras.layers.Layer):
    """A Dense layer that outputs a normally distributed vector."""

    def __init__(self, output_dim, epsilon_std=1., **kwargs):
        self.output_dim = output_dim
        self.epsilon_std = epsilon_std
        super(VariationalLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.z_mean_kernel = self.add_weight(
            shape=(input_shape[1], self.output_dim),
            initializer='glorot_normal',
            trainable=True)
        self.z_mean_bias = self.add_weight(
            shape=(self.output_dim,),
            initializer='zero',
            trainable=True)
        self.z_log_var_kernel = self.add_weight(
            shape=(input_shape[1], self.output_dim),
            initializer='glorot_normal',
            trainable=True)
        self.z_log_var_bias = self.add_weight(
            shape=(self.output_dim,),
            initializer='zero',
            trainable=True)

        super(VariationalLayer, self).build(input_shape)

    def call(self, x):
        z_mean = K.dot(x, self.z_mean_kernel) + self.z_mean_bias
        z_log_var = K.dot(x, self.z_log_var_kernel) + self.z_log_var_bias
        epsilon = K.random_normal(shape=K.shape(z_log_var),
                                  mean=0.,
                                  stddev=self.epsilon_std)

        # Computes variational loss.
        kl_inside = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        self.kl_loss = -0.5 * K.sum(kl_inside, axis=-1)

        # Samples from the distribution to get the output tensor.
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def loss(self, variational_weight=1.):
        """A loss function that can be used by a Keras model."""

        loss = keras.losses.mean_squared_error

        def variational_loss(x, x_rec):
            return loss(x, x_rec) + self.kl_loss * variational_weight

        return variational_loss

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'epsilon_std': self.epsilon_std,
            'loss': self
        }
        base_config = super(VariationalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

This layer calculates the KL divergence loss for us, and provides a loss function that combines it with mean squared reconstruction error. Let's train this on some toy data to visualize what it is doing. The toy data consists of one-hot encoded vectors for the numbers 0 through 9. The encoder will project them down to a two-dimensional latent vector space, then the decoder will map them back up to the original 10-dimensional space. The model will minimize both the mean squared error between the original data and it's reconstruction, and the KL divergence between the latent vector and a normal distribution.


```python
NUM_LATENT_DIMS = 2
variational = VariationalLayer(NUM_LATENT_DIMS)

# Builds the encoder model.
input_var = keras.layers.Input(shape=(10,))
x = keras.layers.Dense(20, activation='tanh')(input_var)
x = variational(x)
encoder = keras.models.Model(input_var, x)

# Builds the decoder model.
input_var = keras.layers.Input(shape=(NUM_LATENT_DIMS,))
x = keras.layers.Dense(20, activation='tanh')(input_var)
x = keras.layers.Dense(20, activation='tanh')(x)
x = keras.layers.Dense(10)(x)
decoder = keras.models.Model(input_var, x)

# Builds the trainable model.
input_var = keras.layers.Input(shape=(10,))
trainable_model = keras.models.Model(inputs=input_var,
                                     outputs=decoder(encoder(input_var)))
trainable_model.compile(loss=variational.loss(1e-2),
                        optimizer='adam',
                        metrics=['accuracy'])

# The toy data will be random one-hot encoded values.
idxs = np.random.randint(0, 10, size=(10000))
toy_distribution = np.eye(10)[idxs]

# To train the autoencoder, train distribution -> distribution.
trainable_model.fit(toy_distribution, toy_distribution,
                    epochs=10, verbose=2, batch_size=32)
```


We can pass data through the encoder to see where it ends up in latent vector space. Because our latent vectors are two-dimensional, we can plot them in regular Cartesian coordinates.


```python
# Uses the decoder to sample from the data distribution.
idxs = np.random.randint(0, 10, size=(300))
inputs = np.eye(10)[idxs]
preds = encoder.predict(inputs)

# Maps the latent space to the predict outputs.
plt.figure(figsize=(10, 10))
for i in range(10):
    ith_preds = preds[idxs == i]
    plt.scatter(ith_preds[:, 0], ith_preds[:, 1], label='pred=%d' % i)
plt.legend(loc=2)
plt.title('Characterizing a two-dimensional latent space')
plt.show()
```

{% include image.html description="Diagram showing the latent space of an autoencoder that learns to encode the numbers 0 through 9 in a two-dimensional vector space. The autoencoder learns to represent each number in it's own part of the latent space, as a normal distribution. More training would reduce the variance of this distribution. Also, note that the entire latent space is approximately normally distributed, thanks to our KL divergence penalty." url="/resources/calcium_autoencoder/autoencoder_7_0.png" %}

As we hoped, the latent dimension clusters the data according to what we want to reconstruct. Now that we've got a variational layer, we can build our model for the calcium data. This model will be a bit more complicated than the one from above, but fundamentally basically the same. We'll use a recurrent neural network, like we did with the classifier.


```python
def check_built(f):
    """A simple wrapper that checks if the model is built, and if not builds it."""
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_built') or not self._built:
            self.build()
            self._built = True
        return f(self, *args, **kwargs)
    return wrapper


def check_calcium_spikes(f):
    """A simple wrapper that checks the calcium and spike inputs."""
    def wrapper(self, calcium, *args, **kwargs):
        assert np.ndim(calcium) == 2
        # Checks that the calcium data is correct.
        if calcium.shape[1] != self.num_input:
            raise ValueError('This autoencoder expects data with '
                             '%d time bins; got %d time bins.'
                             % (self.num_input, calcium.shape[1]))
        return f(self, calcium, *args, **kwargs)
    return wrapper


class VariationalAutoencoder(object):
    """A Variational Autoencoder to approximate the spikefinder data."""

    def __init__(self, num_input, num_latent=256):
        """Initializes the autoencoder parameters.
        Args:
            num_input: int, the number of dimensions in the dataset
                you are trying to approximate.
            num_latent: int, the number of latent dimensions to
                map the data to.
        """
        self._num_input = num_input
        self._num_latent = num_latent

        # Initializes placeholders for the encoder and decoder models.
        self._encoder, self._decoder, self._encoder_decoder = None, None, None

    @property
    def num_input(self):
        return self._num_input

    @property
    def num_latent(self):
        return self._num_latent

    def build(self, encoder_dims=[128, 128], decoder_dims=[128, 128]):
        """Builds the encoder and decoder models."""
        input_calcium = keras.layers.Input(shape=(self.num_input,), name='input_calcium')
        latent_vec = keras.layers.Input(shape=(self.num_latent,), name='latent_vec')
        variational_layer = VariationalLayer(self.num_latent)

        # Builds the encoder.
        calcium_rep = keras.layers.Reshape((self.num_input, 1))(input_calcium)
        calcium_delta = utils.DeltaFeature()(calcium_rep)
        calcium_quad = utils.QuadFeature()(calcium_rep)
        calcium_delta_quad = utils.QuadFeature()(calcium_delta)
        x = keras.layers.Concatenate()([calcium_rep, calcium_delta, calcium_quad])
        x = keras.layers.LSTM(64, return_sequences=False)(x)
        variational_output = variational_layer(x)
        self._encoder = keras.models.Model(inputs=[input_calcium],
                                           outputs=variational_output)

        # Builds the decoder.
        x = latent_vec
        x = keras.layers.RepeatVector(self.num_input)(x)
        x = keras.layers.LSTM(64, return_sequences=True)(x)
        x = keras.layers.Dense(1)(x)
        calcium_pred = keras.layers.Reshape((self.num_input,))(x)
        self._decoder = keras.models.Model(inputs=[latent_vec],
                                           outputs=[calcium_pred])

        output_calcium = self._decoder([variational_output])
        self._encoder_decoder = keras.models.Model(inputs=[input_calcium],
                                                   outputs=[output_calcium])
        self._encoder_decoder.compile(loss=variational_layer.loss(1e-4),
                                      optimizer='adam',
                                      metrics={'did_spike': 'accuracy'})

    @check_built
    @check_calcium_spikes
    def encode(self, calcium):
        """Encodes a sample to get the latent vector associated with it.
        Args:
            calcium: 2D Numpy array with shape (sample_dim, num_time_bins), the
                calcium fluorescence data.
        """
        return self._encoder.predict([calcium])

    @check_built
    def sample(self, num_samples=None, latent_vec=None):
        """Produces samples from the model by feeding the decoder a random vector.
        Args:
            num_samples: int (default: None), number of samples to produce
                from the model, if .
            latent_vec: a 2D Numpy array to use instead of generating a new one.
        Returns:
            calcium_pred: 2D Numpy array with shape (num_samples, num_inputs),
                the predicted calcium trace.
        """
        if latent_vec is None:
            assert num_samples is not None, 'Must specifiy `num_samples`.'
            latent_vec = np.random.normal(loc=0., scale=1.,
                                          size=(num_samples, self.num_latent))
        else:
            num_samples = latent_vec.shape[0]

        calcium_pred = self._decoder.predict([latent_vec])
        return calcium_pred

    @check_built
    @check_calcium_spikes
    def predict(self, calcium):
        """Gets model predictions on some input data.
        Args:
            calcium: 2D Numpy array with shape (sample_dim, num_time_bins), the
                calcium fluorescence data.
        """
        return self._encoder_decoder.predict([calcium])

    @check_built
    @check_calcium_spikes
    def train(self, calcium, epochs=10):
        """Trains the model on some input data.
        Args:
            calcium: 2D Numpy array with shape (sample_dim, num_time_bins), the
                calcium fluorescence data.
            epochs: int, number of training epochs.
        """
        # Trains the encoder-decoder on the data.
        for i in xrange(1, epochs + 1):
            sys.stdout.write('progress: [' + '.' * i + ' ' * (epochs - i) + ']\r')
            sys.stdout.flush()
            self._encoder_decoder.fit([calcium], [calcium],
                                      epochs=1, batch_size=32, verbose=0)
```

Again, we'll only look at a subset of the data. For more completeness it would be a good idea to look at the entire dataset, but a subset of 1000 samples is enough to see the relevant trends.


```python
NUM_VAR_TRAIN = 1000

calcium, did_spike = utils.load_dataset()
num_dimensions = calcium.shape[1]

def get_evenly_split_dataset(num_samples):
    """Gets an evenly-split sample of the data."""
    calcium, did_spike = utils.load_dataset()
    spike_idxs = np.arange(calcium.shape[0])[did_spike == 1]
    nospike_idxs = np.arange(calcium.shape[0])[did_spike == 0]
    spike_idxs = np.random.choice(spike_idxs, num_samples // 2)
    nospike_idxs = np.random.choice(nospike_idxs, num_samples // 2)
    idxs = np.concatenate([spike_idxs, nospike_idxs])
    return calcium[idxs], did_spike[idxs]

# Gets a subset of the data..
calcium, did_spike = get_evenly_split_dataset(NUM_VAR_TRAIN)

model = VariationalAutoencoder(num_dimensions)
model.train(calcium, epochs=100)
```

Now that we've got a trained model, we can visualize how well the model reconstructs the data that we give it.


```python
calcium, did_spike = calcium[:5], did_spike[:5]
pred_calcium = model.predict(calcium)

# Plot the data where no spike was observed on the left,
# and the data where a spike was observed on the right.
fig = plt.figure(figsize=(7, 9))
d = calcium.shape[1] / 2
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    utils.plot_sample(calcium=calcium[i],
                      t_start=-d,
                      t_end=d - 1,
                      sampling_rate=1)
    plt.title('Sample %d, provided data' % i)
    plt.subplot(5, 2, 2 * i + 2);
    utils.plot_sample(calcium=pred_calcium[i],
                      t_start=-d,
                      t_end=d - 1,
                      sampling_rate=1)
    plt.title('Sample %d, reconstruction' % i)

plt.tight_layout()
plt.show()
```

{% include image.html description="Samples from the training data and their reconstruction. The autoencoder does a pretty good job, but smoothes out some of the peaky parts." url="/resources/calcium_autoencoder/autoencoder_13_0.png" %}

These representations are pretty good. Next, let's do PCA on the encoded vectors corresponding to a bunch of data.


```python
# Encodes some samples in latent vector space.
NUM_PCA = 1000
calcium, did_spike = get_evenly_split_dataset(NUM_PCA)

# Encodes data into (NUM_PCA, num_latent)-dimensional array.
latent_vecs = model.encode(calcium)

# Performs PCA on the latent vectors.
pca = PCA(n_components=10, copy=False)
pca.fit(latent_vecs)

# Prints explained variance.
print('Explained variance:')
for i, var in enumerate(pca.explained_variance_ratio_):
    print('    Principal component %d: %.3f' % (i, var))

# Plots embedded latent vectors in 2D space.
pca_vecs = pca.transform(latent_vecs)
x0, x1 = pca_vecs[:, 0], pca_vecs[:, 1]
plt.figure(figsize=(8, 8))
plt.plot(x0[did_spike == 0], x1[did_spike == 0], 'ob')
plt.plot(x0[did_spike == 1], x1[did_spike == 1], 'or')
plt.xlabel('Principal component 0')
plt.ylabel('Principal component 1')
plt.title('Latent vectors for spiking (red) and non-spiking (blue) calcium fluorescences')
plt.show()
```

From doing this analysis, we find that 22% of the variance is captured by the first principle component, and 15% by the second principle component.

    Explained variance:
        Principal component 0: 0.222
        Principal component 1: 0.148
        Principal component 2: 0.086
        Principal component 3: 0.064
        Principal component 4: 0.052
        Principal component 5: 0.040
        Principal component 6: 0.019
        Principal component 7: 0.016
        Principal component 8: 0.011
        Principal component 9: 0.009


We can map the latent vectors to the first two principle component spaces and plot these like we did with the toy example.


{% include image.html description="The completely unsupervised model learns a representation of the data that is fairly well separated into spiking and non-spiking parts along the first principle component. This means that the most *important* components of the calcium fluorescences, in the sense that the autoencoder cares about, are byproducts of spiking, even though the autoencoder has no knowledge of when spiking takes place. However, the first principle component explains only 22% of the variance in the latent space; this means there is a lot of variance coming from other sources besides spiking (noise, basically)." url="/resources/calcium_autoencoder/autoencoder_15_1.png" %}

Because the data is mostly split along the first principle component, this means that spiking (according to our variational autoencoder model) causes more variance than other noisy artifacts. This is a pretty weak conclusion (we expected as much, because calcium imaging is supposed to correlate with spiking), but it is still a good sanity check that this shows up in our model. We can visualize this by simply plotting a histogram of the first principle component.


```python
nbins = 10
plt.figure()
plt.hist(x0[did_spike == 0], nbins, color='blue', histtype='step', stacked=True, fill=False)
plt.hist(x0[did_spike == 1], nbins, color='red', histtype='step', stacked=True, fill=False)
plt.show()
```

{% include image.html description="The latent vectors corresponding spiking (red) and non-spiking (blue) calcium fluorescences, mapped to the first principle component and binned. Even without any supervised learning, the model learns a representation that seems like it could be a good input for a linear classifier." url="/resources/calcium_autoencoder/autoencoder_17_0.png" %}

Finally, let's build a simple Support Vector Machine model to see if we can linearly classify the calcium fluorescences simply based on their latent vector representations.

```python
# Fits a support vector classifier to the model.
clf = svm.SVC(kernel='linear')
clf.fit(latent_vecs, did_spike)

# Computes the predictions and creates a confusion matrix.
preds = clf.predict(latent_vecs)
p_s, p_n = preds[did_spike == 1], preds[did_spike == 0]
n_total = calcium.shape[0]

# Computes confusion matrix values.
ss, ns = np.sum(p_s > 0.5) / n_total, np.sum(p_s <= 0.5) / n_total
sn, nn = np.sum(p_n > 0.5) / n_total, np.sum(p_n <= 0.5) / n_total

print('                     spike    no spike')
print('predicted spike    | %.3f  | %.3f' % (ss, ns))
print('predicted no spike | %.3f  | %.3f' % (sn, nn))
```

                         spike    no spike
    predicted spike    | 0.407  | 0.093
    predicted no spike | 0.076  | 0.424

Wow! Our totally unsupervised learning approach, followed by a linear classifier, was able to slightly out-perform the supervised learning classifier that we created before\*. Fundamentally, the autoencoder extracted features from the data, and the linear classifier used those features to predict (to a reasonable level of accuracy) whether or not those features were relevant for spiking. Another cool bit about using an autoencoder is that we can actually generate samples from the model. Let's sample from the model, providing the decoder with latent vectors that the linear classifier thinks are most strongly correlated with spiking and non-spiking behavior.

<sub>Disclaimer: 95% of the time, this type of approach won't work. This result happened because the unsupervised model was trained for a lot longer than the supervised model.</sub>

```python
%matplotlib inline

# The linear classifier uses a weight vector that is the
# same size as the latent vector. We can feed this weight
# vector and it's negative into the decoder to produce
# samples that would be strongly correlated with "spiking"
# and "non-spiking".

weight_vec = clf.coef_
anti_weight_vec = -weight_vec
cal_corr = model.sample(latent_vec=weight_vec)
cal_anticorr = model.sample(latent_vec=anti_weight_vec)

d = calcium.shape[1] / 2
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
utils.plot_sample(calcium=cal_corr[0],
                  t_start=-d,
                  t_end=d - 1,
                  sampling_rate=1)
plt.title('Strongly correlated with spiking')
plt.subplot(1, 2, 2)
utils.plot_sample(calcium=cal_anticorr[0],
                  t_start=-d,
                  t_end=d - 1,
                  sampling_rate=1)
plt.title('Strongly anti-correlated with spiking')
plt.tight_layout()
plt.show()
```

{% include image.html description="The outputs of the decoder when it is fed the latent vector that the linear classifier thinks are most strongly correlated with spiking (left) and the latent vector most strongly anticorrelated with spiking (right). The model seems to think that strong depression 8 milliseconds in the past is a good indicator for a spike, and a strong depreseion right at the current time is a good indicator for no spike." url="/resources/calcium_autoencoder/autoencoder_23_0.png" %}

## Acknowledgements

[David Nicholson](https://nickledave.github.io/) wrote the abstract and was the reason I put this tutorial together. The data was collected by [Lucas Theis](http://theis.io/) for the [SpikeFinder competition](http://spikefinder.codeneuro.org/).
