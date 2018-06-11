---
layout: post
title: "Decoding Gibberish using Neural Networks"
date: 2017-04-16 12:00:00
categories: deep-learning language visualizations
excerpt: >
  Training a neural network to predict word embeddings from spelling, and using
  nearest neighbor search to decode meaning.
links:
 - View Code: https://github.com/codekansas/gibberish-decoder
---

<div class="panel panel-default">
<div class="panel-heading">
<div class="form-inline">
  <input type="text" class="form-control" placeholder="" aria-label="" aria-describedby="basic-addon1" id="word">
  <button type="submit" class="btn btn-primary mb-2" id="search-button">Submit</button>
</div>
</div>
<div class="panel-body" id="word-display">
The nearest neighbor words will show up here.
</div>
</div>

This is a simple experiment in predicting a word's meaning purely from it's spelling. A recurrent neural network is first trained to predict word embeddings from spelling, from a subset of 2000 word-embedding pairs [trained using Glove on a Wikipedia crawl](https://nlp.stanford.edu/projects/glove/). This subset is filtered for common words that are at least five letters long, so a lot of words won't appear in the search results. Next, a new word is fed to the network to get it's word embedding. The nearest neighbors to that embedding are shown; these represent the words that the network thinks are closest in meaning to the presented word. This happens entirely in your browser using <a href="https://github.com/transcranial/keras-js">Keras JS</a>. Nearest neighbor decoding is done using [NumJS](https://github.com/nicolaspanel/numjs).

## Neural Networks in your Browser

Because the model is deployed using <a href="https://github.com/transcranial/keras-js">Keras JS</a>, you can run a full neural network in your browser, despite my site being hosted statically on [Github Pages](https://pages.github.com/) (although the network files are pretty large, so it can be kind of buggy, especially on slow connections). In the future I hope to make more interactive examples like this, for language modeling and other things.

{% include image.html url='/assets/pics/posts/gibberish/encoder_network.png' description='Diagram of the embedding model structure' %}

## Nearest Neighbor Decoding

[Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding) are a ubiquitous idea in language modeling that suggests that we can represent a word as a vector, where the location of the vector in vector space represents the meaning of that word. There are some [cool examples](https://www.quora.com/What-are-some-interesting-Word2Vec-results) of what this looks like in practice. The most common example is that if you take the vector for "king", subtract the vector for "man" and add the vector for "woman" you get the vector for "queen". This means that our neural network is essentially trying to predict the word's meaning purely from it's spelling. We can decode it's meaning by finding words with similar vectors. As with a lot of neural applications, interpreting the results is a bit like finding images in clouds.

{% include image.html url='/assets/pics/posts/gibberish/vector_embeddings.png' description='Visualization of words in vector space' %}

<script defer src="{{ site.cdn.mathjs }}"></script>
<script defer src="{{ site.cdn.kerasjs-legacy }}"></script>
<script defer src="/assets/posts/gibberish/main.js"></script>
