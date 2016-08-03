---
layout: post
title: "Using Gensim Word2Vec Embeddings in Keras"
date: 2016-08-02 12:00:00
categories: machine-learning
excerpt: >
  A short post and script regarding using Gensim Word2Vec embeddings in Keras.
---

<table class="note">
<tr><th>Note</th></tr>
<tr><td>
The full script for generating and testing word embeddings, as described in this post, can be found <a href="/resources/embeddings/embeddings.txt" target="_blank">here</a>.
</td></tr>
</table>

* TOC
{:toc}

# Introduction

This will be a quick post about using Gensim's Word2Vec embeddings in Keras. This topic has been covered elsewhere by other people, but I thought another code example and explanation might be useful.

# Resources

 - [Keras Blog][keras-blog]: Francois Chollet wrote a whole post about this exact topic a few weeks ago, so that's the authoritative source on how to do this.
 - [Github Issue][github-issue]: Another reference, with some relevant code.
 - [Discussion on the Google Group][google-group-discussion]: This topic was hashed out about a year ago on the Keras Google Group, and has since migrated to its own Slack channel.

# Installing Dependencies

Usually `pip install ...` works if you don't already have Keras or Gensim.

{% highlight bash %}
sudo pip install Theano
sudo pip install keras
sudo pip install --ignore-installed gensim
{% endhighlight %}

# Tokenizing

{% highlight python %}
from gensim.utils import simple_preprocess
tokenize = lambda x: simple_preprocess(x)
{% endhighlight %}

*In lexical analysis, tokenization is the process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens. -Wikipedia*

We want to tokenize each string to get a list of words, usually by making everything lowercase and splitting along the spaces. In contrast, *lemmatization* involves getting the root of each word, which can be helpful but is more computationally expensive (enough so that you would want to preprocess your text rather than do it on-the-fly).

# Create Embeddings

{% highlight python %}
import os
import json
import numpy as np
from gensim.models import Word2Vec

def create_embeddings(data_dir, embeddings_path, vocab_path, **params):
    class SentenceGenerator(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            for fname in os.listdir(self.dirname):
                for line in open(os.path.join(self.dirname, fname)):
                    yield tokenize(line)

    sentences = SentenceGenerator(data_dir)

    model = Word2Vec(sentences, **params)
    weights = model.syn0
    np.save(open(embeddings_path, 'wb'), weights)

    vocab = dict([(k, v.index) for k, v in model.vocab.items()])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))
{% endhighlight %}

We first create a `SentenceGenerator` class which will generate our text line-by-line, tokenized. This generator is passed to the Gensim Word2Vec model, which takes care of the training in the background. We can pass parameters through the function to the model as keyword `**params`.

## Key Observation

The `syn0` weight matrix in Gensim corresponds exactly to weights of the `Embedding` layer in Keras. We want to save it so that we can use it later, so we dump it to a file. We also want to save the vocabulary so that we know which columns of the Gensim weight matrix correspond to which word; in Keras, this dictionary will tell us which index to pass to the `Embedding` layer for a given word. We'll dump this as a JSON file to make it more human-readable.

# Loading Vocabulary

{% highlight python %}
import json

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word
{% endhighlight %}

We can load the vocabulary from the JSON file, and generate a reverse mapping (from index to word, so that we can decode an encoded string if we want).

# Loading Embeddings

{% highlight python %}
from keras.layers import Embedding
from keras.engine import Input

def word2vec_embedding_layer(embeddings_path):
    weights = np.load(open(embeddings_path, 'rb'))
    layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])
    return layer

# elsewhere
input_vec = Input(shape=(sentence_length,), dtype='int32', name='input')
embedding = word2vec_embedding_layer('/path/to/save/file')
embedded = embedding(input_vec)
{% endhighlight %}

It turns out to be super straightforward! We just pop the weights from the Gensim model into the Keras layer. We can then use this layer to embed our inputs as we normally would.

# Cosine Similarity

{% highlight python %}
input_a = Input(shape=(1,), dtype='int32', name='input_a')
input_b = Input(shape=(1,), dtype='int32', name='input_b')
embeddings = word2vec_embedding_layer(options.embeddings)
embedded_a = embeddings(input_a)
embedded_b = embeddings(input_b)
similarity = merge([embedded_a, embedded_b], mode='cos', dot_axes=2)
model = Model(input=[input_a, input_b], output=similarity)
{% endhighlight %}

The canonical usage for word embeddings is to see that similar words are near each other. We can measure the cosine similarity between words with a simple model like this (note that we aren't training it, just using it to get the similarity).


[script-link]: /resources/embeddings/embeddings.py
[keras-blog]: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
[github-issue]: https://github.com/fchollet/keras/issues/853
[google-group-discussion]: https://groups.google.com/forum/#!topic/keras-users/4wUnPDutY5o
[my-gist]: https://gist.github.com/codekansas/15b3c2a2e9bc7a3c345138a32e029969

