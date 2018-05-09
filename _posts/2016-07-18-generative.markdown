---
layout: post
title: "The Unreasonable Effectiveness of Restricted Boltzmann Machines"
date: 2016-07-18 12:00:00
categories: deep-learning
keywords:
 - Restricted Boltzmann Machines
 - Time Series Modeling
 - Deep Learning
image: /resources/index/rnnrbm.png
excerpt: >
  Building on the Recurrent RBM for sequence modeling. This post relates to what I am doing for my Master's thesis.
links:
 - View Code: https://github.com/codekansas/generative-modeling
---

# Introduction

For my Master's thesis, I'm working on modeling some time-dependent sequences. There is a pretty rich set of literature associated with doing this, much of it related to addressing the unique challenges posed in voice recognition.

{% include multiimage.html url1="/resources/generative_modeling/rnnrbm.png" url2="/resources/generative_modeling/rnnrbm2.png" url3="/resources/generative_modeling/rnnrbm3.png" description="Samples from the RNN-RBM applied to spectrograms of animal vocalizations." %}

## Terminology

In order to understand the details of this post, it would be good to familiarize yourself with the following concepts, which will be touched on throughout the post:

 - **Restricted Boltzmann Machines**: An energy-based model, meaning that it takes an input dataset and "reduces its energy". Conceptually, an RBM is "run" by alternating back and fourth between visible and hidden neurons; after alternating back and fourth for a while, the model will settle to a low-energy configuration (ideally our dataset). For a mathematical explanation, these resources are available:
   - [Implementing an RBM in Theano][deeplearning-rbm]
   - [Wikipedia article on RBMs][wikipedia-rbm]
 - **Recurrent Neural Networks**: These are becoming very popular for modeling a whole host of things. I've previously written about language modeling with RNNs; other applications include computer vision (helping the computer figure out which parts of an image are important) and speech recognition (where temporal dependencies are very important).

## References

Some references which explore this topic in greater detail can be found here:

 - [Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription][modeling-temporal-dependencies]: This paper from the University of Montreal in 2012 presented a Restricted Boltzmann Machine model which was used to generate music. They tied together timesteps of a Restricted Boltzmann Machine with recurrent units, allowing it to generatively model time-varying sequences. They wrote a very nice tutorial on how to build their model [here][deeplearning-rnnrbm].
 - [The Unreasonable Effectiveness of Recurrent Neural Networks][karpathy-effectiveness]: This was a really popular blog post about using neural networks to generate text based on a reference text. There are many cool examples of applying this to other things, including [generating Irish Folk music](https://soundcloud.com/seaandsailor/sets/char-rnn-composes-irish-folk-music) and [speeches like Obama](https://medium.com/@samim/obama-rnn-machine-generated-political-speeches-c8abd18a2ea0#.38lq4zkal).
 - [Speech Recognition with Deep Recurrent Neural Networks][graves-speech]: A pretty influential paper from Geoff Hinton and company in 2013 which explains LSTMs in greater detail.
 - [Multimodal Learning with Deep Boltzmann Machines][multimodal-learning] This paper describes using a probabilistic model to couple data from different modalities (in particular, vision and text) and generate one given the other. For example, given some keywords, their model is able to generate relevant images, and given an image, is able to generate descriptive keywords.

# Installing dependencies

Most of this post will rely on using Theano. The general concepts can probably be ported over to another framework pretty easily (if you do this, I would be interested in hearing about it). It probably also helps to have a GPU, if you want to do more than try toy examples. You can follow the installation instructions [here](http://deeplearning.net/software/theano/install.html), although getting a GPU working with your system can be a bit painful.

# Another take on restricted Boltzmann Machines

The question which RBMs are often used to answer is, "What do we do when we don't have enough labeled data?" Approaching this question from a neural network perspective would probably lead you to the [autoencoder](https://en.wikipedia.org/wiki/Autoencoder), where instead of training a model to produce some output given an input, you train a model to reproduce the input. Autoencoders are easy to think about, because they build on the knowledge that most people have about conventional neural networks. However, in practice, RBMs tend to outperform autoencoders for important tasks. In the example below, both models are trained on the MNIST dataset and learn some filters (the weights connecting the visible vector to a single hidden unit).

{% include image.html description="Comparison of filters learned by an autoencoder and an RBM" url="/resources/generative_modeling/autoencoder_rbm_mnist.png" %}

[Boulanger-Lewandowski, Bengio, and Vincent (2012)][modeling-temporal-dependencies] suggests that unlike a regular discriminative neural network, RBMs are better at modeling multi-modal data. This is evident when comparing the features learned by the RBM on the MNIST task with those learned by the autoencoder; even though the autoencoder did learn some spatially localized features, there aren't very many multi-modal features. In contast, the majority of the features learned by the RBM are multimodal; they actually look like penstrokes, and preserve a lot of the correlated structure in the dataset.

## Formulas

By definition, the connection weights of an RBM define a probability distribution

$$P(v) = \frac{1}{Z} \sum_{h}{e^{-E(v,h)}}$$

Given a piece of data $$\tilde{x}$$, parameters $$\theta$$ are updated to increase the probability of the training data and decrease the probability of samples generated by the model

$$-\frac{\delta \log p(x)}{\delta \theta} = \frac{\delta \mathscr{F}(x)}{\delta \theta} - \sum_{\tilde{x}}{p(\tilde{x})\frac{\delta \mathscr{F}(x)}{\delta \theta}}$$

where $$\mathscr{F}$$ indicates the free energy of a visible vector, or the negative log of the sum of joint energies of that visible vector and all possible hidden vectors

$$\mathscr{F}(x) = -\log \sum_{h}{e^{-E(x,h)}}$$

The explicit derivatives used to update the visible-hidden connections are

$$-\frac{\delta \log p(v)}{\delta W_{ij}} = E_v[p(h_i | v) * v_j] - v_j^{(i)} * sigm(W_i * v^{(i)} + c_i)$$

In words, the connection between a visible unit $$v_j$$ and a hidden unit $$h_i$$ is changed so that the expected activation of that hidden unit goes down in general, but goes up when the data vector is presented, if the visible unit is on in that data vector.

## Intuition

Samples are "generated by the model" by repeatedly jumping back and forth from visible to hidden units. It is not evident why this gives a probability distribution. Suppose you choose a random hidden vector; given the connections between layers, that vector maps to a visible vector. The probability distribution of visible vectors is therefore generated from the hidden distribution. We would like to mold the model so that our random hidden vector will be more likely to map to a visible vector in our dataset. If that doesn't work, we would like to tweek the model so that a random visible vector will map to a hidden vector that maps to our dataset. And so on. After training, running the model on a random probability distribution twists it around to give us a probability distribution of visible vectors that is close to our dataset.

The best way to think about what an RBM is doing during learning is that it is increasing the probability of a good datapoint, then running for a bit to get a bad datapoint, and decreasing its probability. It is changing the probabilities by updating the connections so that the bad datapoint is more likely to map to the good datapoint than the other way around. So when you have a cluster of good datapoints, their probabilities will be increased together (since they are close to each other, they are unlikely to be selected as the "bad" point of another sample in the cluster), and the probability of all the points around that cluster will be decreased. This illustrates the importance of increasing the number of steps of Gibb's sampling as training goes on, in order to get out of the cluster. This also gives some intuition on why RBMs learn multi-modal representations that autoencoders can't; RBMs find clusters of correlated points, while autoencoders only learn representations which minimize the amount of information required to represent some vectors.

# Modeling time-varying statistics

As described above, there is some reason to think that an RBM model may learn higher-order correlations better than a traditional neural network. However, as they are conventionally described, they can't model time-varying statistics very well. For many applications this presents a serious drawback. The top answer on Quora for the question [Are Deep Belief Networks useful for Time Series Forecasting?](https://www.quora.com/Are-Deep-Belief-Networks-useful-for-Time-Series-Forecasting) is by Yoshua Bengio, who suggests looking at the work of his Ph.D. student, Nicolas Boulanger-Lewandowski, who wrote the tutorial that much of this blog post is modeled around. In particular, the paper [Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription][modeling-temporal-dependencies] and it's corresponding [tutorial][deeplearning-rnnrbm] provide a good demonstration of doing almost exactly what Andrej Karpathy's [blog post][karpathy-effectiveness] does, although instead of using RNNs to continually predict the next element of a sequence, it does something a little differrent.

The RNN-RBM uses an RNN to generate a visible and hidden bias vector for an RBM, and then trains the RBM normally (to reduce the energy of the model when initialized with those bias vectors and the visible vector at the first time step). Then the next visible vector is fed into the RNN and RBM, the RNN generated another set of bias vectors, and the RBM reduces the energy of that new configuration. This is repeated for the whole sequence.

What exactly does this training process do? Let's consider the application that is described in both the paper and tutorial, generating polyphonic music (polyphonic here just means there may be multiple notes at the same time step). The weight matrix of the RBM, which has dimensions `<n_visible, n_hidden>`, provides features that activate individual hidden units in response to a particular pattern of visible units. For music, these features are chords, which played at each timestep to generate a song; for video, these features are individual frames, which are very similar to the features learned on the MNIST dataset.

The RNN part is trained to generate biases that activate the right features of the RBM in the right order; in other words, the RNN tries to predict the next set of features given a past set. When we switch the RBM from learning a probability distribution to generating one, the RNN is used to generate biases for the RBM, defining a pattern of activating filters. The stochasticity of the RBM is what gives the model its nondeterminism.


[graves-speech]: http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf
[wikipedia-rbm]: https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine
[deeplearning-rbm]: http://deeplearning.net/tutorial/rbm.html
[deeplearning-rnnrbm]: http://deeplearning.net/tutorial/rnnrbm.html
[karpathy-effectiveness]: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
[modeling-temporal-dependencies]: http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf
[multimodal-learning]: https://papers.nips.cc/paper/4683-multimodal-learning-with-deep-boltzmann-machines.pdf
[github-repo]: https://github.com/codekansas/generative-modeling
