---
layout: post
title: "Deep Language Modeling with Keras"
date: 2016-04-27 12:00:00
categories: ml
---

# Contents

 - [Introduction](#introduction)
 - [Installing Keras](#installing-keras)
 - [Jumping Into Language Modeling](#jumping-in)
   - [Word Embedding](#embedding)
     - [Code Example](#embedding-code-example)
   - [Recurrent Neural Networks](#rnn)
     - [Vanilla RNN update equation](#vanilla)
     - [LSTM update equations](#lstm)
     - [GRU update equations](#gru)
     - [Code Example](#rnn-code-example)
   - [Attentional RNNs](#attentional)
   - [Convoluted Neural Networks](#convolutional)
   - [Similarity Metrics](#cosine)
 - [Closing remarks](#closing)

# <a name="introduction"></a>Introduction

[Question answering][qa wiki] has recieved more focus as large search engines have basically mastered general information retrieval and are starting to cover more edge cases. Question answering happens to be one of those edge cases, because it could involve a lot of syntatic nuance that doesn't get captured by standard information retrieval models, like LDA or LSI. Hypothetically, deep learning models would be better suited to this type of task because of their ability to capture higher-order syntax. Two papers, "Applying deep learning to answer selection: a study and an open task" [(Feng et. al. 2015)][feng] and "LSTM-based deep learning models for non-factoid answer selection" [(Tan et. al. 2016)][tan], are recent examples which have applied deep learning to question-answering tasks with good results.

[Feng et. al.][feng] used an in-house Java framework for their work, and [Tan et. al.][tan] built their model entirely from Theano. Personally, I am a lot lazier than them, and I don't understand CNNs very well, so I would like to use an existing framework to build one of their models to see if I could get similar results. [Keras][keras] is a really popular one that has support for everything we might need to put the model together.

# <a name="installing-keras"></a>Installing Keras

See the instructions [here](http://keras.io/#installation) on how to install Keras. The simple route is to install using `pip`, e.g.

    sudo pip install --upgrade keras

There are some important features that might not be available without the most recent version. I'm not sure if doing `pip install` gets the most recent version, so it might be helpful to install from binary. This is actually pretty straightforward! Just change to the directory where you want your source code to be and do:

    git clone https://github.com/fchollet/keras.git .
    sudo python setup.py install

One benefit of this is that if you want to add a custom layer, you can add it to the Keras installation and be able to use it across different projects.

# <a name="jumping-in"></a>Jumping into language modeling

There are actually a couple language models in the [Keras examples](https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py):

 - [`imdb_lstm.py`](https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py): Using a LSTM recurrent neural network to do sentiment analysis on the IMDB dataset
 - [`imdb_cnn_lstm.py`](https://github.com/fchollet/keras/blob/master/examples/imdb_cnn_lstm.py) The same task, but this time using a CNN layer beneath the LSTM layer
 - [`babi_rnn.py`](https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py): Recurrent neural networks for modeling Facebook's bAbi dataset, "a mixture of 20 tasks for testing text understanding and reasoning"

These are pretty interesting to play around with. It is really cool how easy it is to get one of these set up! With Keras, a high-level model design can be quickly implemented.

## <a name="embedding"></a>Word Embeddings

Ok! Let's dive in. The first challenge that you might think of when designing a language model is what the units of the language might be. A reasonable dataset might have around 20000 distinct words, after lemmatizing them. If the average sentence is 40 words long, then you're left with a `20000 x 40` matrix just to represent one sentence, which is 3.2 megabytes if each word is represented in 32 bits. This obviously doesn't work, so the first step in developing a good language model is to figure out how to reduce the number of dimensions required to represent a word.

One popular method of doing this is using `word2vec`. `word2vec` is a way of embedding words in a vector space so that words that are semantically similar are near each other. There are some interesting consequences of doing this, like being able to do word addition and subtraction:

    king - man + woman = queen

In Keras, this is available as an `Embedding` layer. This layer takes as input a `(n_batches, sentence_length)` dimensional matrix of integers representing each word in the corpus, and outputs a `(n_batches, sentence_length, n_embedding_dims)` dimensional matrix, where the last dimension is the word embedding.

There are two advantages to this. The first is space: Instead of 3.2 megabytes, a 40 word sentence embedded in 100 dimensions would only take 16 kilobytes, which is much more reasonable. More importantly, word embeddings give the model a hint at the meaning of each word, so it will converge more quickly. There are significantly fewer parameters which have to be jostled around, and parameters are sort of tied together in a sensible way so that they jostle in the right direction.

Here's how you would go about writing something like this:

{% highlight python %}
from keras.layers import Input, Embedding

input_sentence = Input(shape=(sentence_maxlen,), dtype='int32')
embedding = Embedding(n_words, n_embed_dims)(input_sentence)
{% endhighlight %}

Let's try this out! We can train a recurrent neural network to predict some dummy data and examine the embedding layer for each vector. This model takes a sentence like "sam is red" or "sarah not green" and predicts what color the person is. It is a very simple example with some dummy data. 

<a name="embedding-code-example"></a>
{% highlight python %}
import itertools
import numpy as np

sentences = '''
sam is red
hannah not red
hannah is green
bob is green
bob not red
sam not green
sarah is red
sarah not green'''.strip().split('\n')
is_green = np.asarray([[0, 1, 1, 1, 1, 0, 0, 0]], dtype='int32').T

lemma = lambda x: x.strip().lower().split(' ')
sentences_lemmatized = [lemma(sentence) for sentence in sentences]
words = set(itertools.chain(*sentences_lemmatized))
# set(['boy', 'fed', 'ate', 'cat', 'kicked', 'hat'])

# dictionaries for converting words to integers and vice versa
word2idx = dict((v, i) for i, v in enumerate(words))
idx2word = list(words)

# convert the sentences a numpy array
to_idx = lambda x: [word2idx[word] for word in x]
sentences_idx = [to_idx(sentence) for sentence in sentences_lemmatized]
sentences_array = np.asarray(sentences_idx, dtype='int32')

# parameters for the model
sentence_maxlen = 3
n_words = len(words)
n_embed_dims = 3

# put together a model to predict 
from keras.layers import Input, Embedding, merge, Flatten, RNN
from keras.models import Model

input_sentence = Input(shape=(sentence_maxlen,), dtype='int32')
input_embedding = Embedding(n_words, n_embed_dims)(input_sentence)
color_prediction = RNN(1)(input_embedding)

predict_green = Model(input=[input_sentence], output=[color_prediction])
predict_green.compile(optimizer='sgd', loss='binary_crossentropy')

# fit the model to predict what color each person is
predict_green.fit([sentences_array], [is_green], nb_epoch=5000, verbose=1)
embeddings = predict_green.layers[1].W.get_value()

# print out the embedding vector associated with each word
for i in range(n_words):
	print('{}: {}'.format(idx2word[i], embeddings[i]))
{% endhighlight %}

The embedding layer embeds the words into 3 dimensions. A sample of the vectors it produces is seen below. As predicted, the model learns useful word embeddings.

    sarah:	[-0.5835458  -0.2772688   0.01127077]
    sam:	[-0.57449967 -0.26132962  0.04002968]

    bob:	[ 1.10480607  0.97720605  0.10953052]
    hannah:	[ 1.12466967  0.95199704  0.13520472]

    not:	[-0.17611612 -0.2958962  -0.06028322]
    is:	[-0.10752882 -0.34842652 -0.06909169]

    red:	[-0.10381682 -0.31055665 -0.0975003 ]
    green:	[-0.05930901 -0.33241618 -0.06948926]

Each category is grouped in the 3-dimensional vector space. The network learned each of these categories from how each word was used. This is very useful for developing a language model.

## <a name="rnn"></a>Recurrent Neural Networks

As the Keras examples illustrate, there are different philosophies on deep language modeling. [Feng et. al.][feng] did a bunch of benchmarks with convolutional networks, and ended up with some impressive results. [Tan et. al.][tan] used recurrent networks with some different parameters. I'll focus on recurrent neural networks (What do pirates call neural networks? *Arrrgh*NNs) first. In general, I'll assume some familiarity with both recurrent and convolutional neural networks. [Andrej Karpathy's blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) discusses neural networks in detail. Here is an image from that post which explains the core concept:

![Recurrent neural network](/images/karpathy_rnn.jpeg)

### <a name="vanilla"></a>Vanilla

The basic RNN architecture is essentially a feed-forward neural network that is stretched out over a bunch of time steps and has it's intermediate output added to the next input step. This idea can be expressed as an update equation for each input step:

    new_hidden_state = tanh(dot(input_vector, W) + dot(prev_hidden, U) + b)

Note that `dot` indicates vector-matrix multiplication. Multiplying a vector of dimensions `<m>` by a matrix of dimensions `<m, n>` can be done with `dot(<m>, <m, n>)` and yields a vector of dimensions `<n>`. This is consistent with its usage in Theano and Keras. In the update equation, we multiply each `input_vector` by our input weights `W`, multiply the `prev_hidden` vector by our hidden weights `U`, and add a bias, before passing the sum to the activation function `sigmoid`. To get the **many to one** behavior in the image, we can grab the last hidden state and use that as our output. To get the **one to many** behavior, we can pass one input vector and then just pass a bunch of zero vectors to get as many hidden states as we want.

### <a name="lstm"></a>LSTM

If the RNN gets really long, then we run into a lot of difficulty training the model. The effect of something a early in the sequence on the end result is very small relative to later components, so it is hard to use that information in updating the weights. To solve this, several methods have been proposed, and two have been implemented in Keras. The first is the Long Short-Term Memory (LSTM) unit, which was proposed by [Hochreiter and Schmidhuber 1997][hochreiter]. This model uses a second hidden state which stores information from further back in the model, allowing that information to have a stronger effect on the end result. The update equations for this model are:

    input_gate = tanh(dot(input_vector, W_input) + dot(prev_hidden, U_input) + b_input)
    forget_gate = tanh(dot(input_vector, W_forget) + dot(prev_hidden, U_forget) + b_forget)
    output_gate = tanh(dot(input_vector, W_output) + dot(prev_hidden, U_output) + b_output)

    candidate_state = tanh(dot(x, W_hidden) + dot(prev_hidden, U_hidden) + b_hidden)
    memory_unit = prev_candidate_state * forget_gate + candidate_state * input_gate

    new_hidden_state = tanh(memory_unit) * output_gate

Note that `*` indicates element-wise multiplication. This is consistent with its usage in Theano and Keras. First, there are a bunch more parameters to train; not only do we have weights for the input-to-hidden and hidden-to-hidden matrices, but also we have an accompanying `candidate_state`. The candidate state is like a second hidden state that transfers information to and from the hidden state. It is like a safety deposit box for putting things in and taking things out.

### <a name="gru"></a>GRU

The second model is the Gated Recurrent Unit (GRU), which was proposed by [Cho et. al. 2014][cho]. The equations for this model are as follows:

    update_gate = tanh(dot(input_vector, W_update) + dot(prev_hidden, U_update) + b_update)
    reset_gate = tanh(dot(input_vector, W_reset) + dot(prev_hidden, U_reset) + b_reset)

    reset_hidden = prev_hidden * reset_gate
    temp_state = tanh(dot(input_vector, W_hidden) + dot(reset_hidden, U_reset) + b_hidden)
    new_hidden_state = (1 - update_gate) * temp_state + update_gate * prev_hidden

In this model, there is an `update_gate` which controls how much of the previous hidden state to carry over to the new hidden state and a `reset_gate` which controls how much the previous hidden state changes. This allows potentially long-term dependencies to be propagated through the network.

My implementations of these models in Theano, as well as optimizers for training them, can be found in [this Github repository][theano-rnn].

Now that we've seen the equations, let's see how Keras implementations compare on some sample data.

<a name="rnn-code-example"></a>
{% highlight python %}
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
{% endhighlight %}

Training on my computer, there wasn't a significant difference between the LSTM and GRU layers. However, the Vanilla layer had a much more difficult time learning the data. This makes sense! There are 10 different sequences of random 0s and 1s that need to be learned. The probability of two sequences being different doubles with each extra element in the sequence, so a model that can take advantage of long-term dependencies will have a much easier time seeing how two sequences are different.

## <a name="attentional"></a>Attentional RNNs

Add a part about attention component here.

## <a name="convolutional"></a>Convolutional Neural Networks

I'll add this eventually, I think. Right now I should probably go do something else for a bit.

## <a name="cosine"></a>Similarity Metrics

The basic idea with question answering is to embed questions and answers as vectors, so that the question vector is close in vector space to the answer vector. "Close" usually means it has a small cosine distance.

## <a name="closing"></a>Closing remarks

This post follows my final project for my Information Retrieval class, the code for which can be seen [here][github project].


[theano-rnn]: https://github.com/codekansas/theano-rnn
[github project]: https://github.com/codekansas/keras-language-modeling
[qa wiki]: https://en.wikipedia.org/wiki/Question_answering
[hochreiter]: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
[cho]: http://arxiv.org/pdf/1406.1078.pdf
[feng]: http://arxiv.org/pdf/1508.01585v2.pdf
[tan]: http://arxiv.org/pdf/1511.04108.pdf
[keras]: https://github.com/fchollet/keras
[karpathy]: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
