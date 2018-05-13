---
layout: post
title: ""
date: 2018-05-08 12:00:00
categories: deep-learning
keywords:
 - GANs
excerpt: >
  A tutorial on how to quickly code up various types of GANs in Keras.
---

# Introduction

[Generative Adversarial Networks][gan-paper] took the deep learning world by storm a few years ago, thanks to their remarkable ability to model data distributions and the elegant way that they slot into the deep neural network way of thinking. [Keras][keras-github] is the most popular "comprehensive" deep learning system, and has been used with excellent results on many Kaggle competitions. However, it can take a little bit of hacking to implement GANs in Keras, because they aren't *minimizing a loss function*. In this post, I'll outline how to quickly build different types of GANs in Keras and the required hacks (specifically, for the version of Keras that ships with Tensorflow 1.8). All of these examples will be implemented on MNIST, so you won't need a big GPU and lots of time at home to replicate them.

# Models

I'll use a few basic Keras models throughout this post. If you want to just copy-and-paste them that works. Here's the way I typically import things, but I try to make the code below ambiguous enough that you can use whatever backend you like.

```python
import tensorflow as tf
from tensorflow import keras as ks
K = ks.backend
```

## Generator

The generator that I'm going to be using has a latent dimension of 100, and consists of a series of upsampling convolutions with ReLUs. This is copped from the [ACGAN in the Keras examples][keras-examples-acgan].

```python
def get_generator_model(latent_dim=100):
    '''Creates a generator model for the MNIST dataset.
    Args:
        latent_dim (int): the number of dimensions in the latent vector.
    Returns:
        model with input shape (latent_dim) and output shape (28, 28, 1).
    '''
    return ks.models.Sequential([
        ks.layers.Dense(3 * 3 * 384, input_dim=latent_dim, activation='relu'),
        ks.layers.Reshape((3, 3, 384)),
        ks.layers.Conv2DTranspose(192, 5, strides=1, padding='valid',
                                  activation='relu',
                                  kernel_initializer='glorot_normal'),
        ks.layers.BatchNormalization(),
        ks.layers.Conv2DTranspose(96, 5, strides=2, padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_normal'),
        ks.layers.BatchNormalization(),
        ks.layers.Conv2DTranspose(1, 5, strides=2, padding='same',
                                  activation='tanh',
                                  kernel_initializer='glorot_normal'),
        ks.layers.Activation('tanh'),
    ])
```

### Auxiliary Classifier Generator

The above "generator" can easily be adapted for an Auxiliary Classifier GAN (ACGAN). Note that, in this version, we don't explicitly provide the latent variable to the model; to do that, you can use the functional API, as demonstrated in the [canonical example][keras-examples-acgan].

```python
def get_ac_generator_model(latent_dim=100,
                           latent_input=False,
                           generator_model=None):
    '''Creates an auxiliary classifier generator model for the MNIST dataset.
    Args:
        latent_dim (int): the number of dimensions in the latent vector.
        latent_input (bool): if set, adds a separate
        generator_model (keras model): if provided, the model to use as the
            generator (for example, with pre-trained weights).
    Returns:
        model with input shape (10) (the integer class) and, if
        latent_input=True, (latent_dim), and output shape (28, 28, 1).
    '''
    generator_model = generator_model or get_generator_model(latent_dim)

    input_class = ks.layers.Input(shape=(num_classes,))
    model_inputs = [input_class]

    x = ks.layers.Embedding(num_classes, latent_dim,
                            embeddings_initializer='glorot_normal')
    x = ks.layers.Flatten()(x)

    if latent_input:
        latent_mult = ks.layers.Input(shape=(latent_dim,))
        model_inputs.append(latent_mult)
        x = ks.layers.Lambda(lambda x: x[0] * x[1])([x, latent_mult])
    else:
        x = ks.layers.Lambda(lambda x: x * K.random_normal(K.shape(x)))

    generated_image = generator_model(x)
    return ks.models.Model(inputs=model_inputs, outputs=[generated_image])
```

## Discriminator

The discriminator model (and other models down the line) rely on a "base" model that downsamples the input to a flattened embedding layer.

```python
def get_downsample_model():
    return ks.models.Sequential([
        ks.layers.Conv2D(32, 3, padding='same', strides=2,
                         input_shape=(28, 28, 1)))
        ks.layers.LeakyReLU(0.2),
        ks.layers.Dropout(0.3),
        ks.layers.Conv2D(64, 3, padding='same', strides=1),
        ks.layers.LeakyReLU(0.2),
        ks.layers.Dropout(0.3),
        ks.layers.Conv2D(128, 3, padding='same', strides=2),
        ks.layers.LeakyReLU(0.2),
        ks.layers.Dropout(0.3),
        ks.layers.Conv2D(256, 3, padding='same', strides=1),
        ks.layers.LeakyReLU(0.2),
        ks.layers.Dropout(0.3),
        ks.layers.Flatten(),
    ])
```

A basic discriminator can be made by

[gan-paper]: http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
[keras-github]: https://github.com/keras-team/keras
[keras-examples-acgan]: https://github.com/keras-team/keras/blob/master/examples/mnist_acgan.py
[acgan-paper]: https://arxiv.org/abs/1610.09585
