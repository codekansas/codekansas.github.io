---
layout: post
title: "Decoding Gibberish using Neural Networks"
date: 2017-04-16 12:00:00
categories: machine-learning
keywords:
 - Word Embeddings
 - Glove
 - Keras
excerpt: >
  Training a neural network to predict word embeddings from spelling, and using
  nearest neighbor search to decode meaning.
image: /resources/index/vector_embeddings.png
links:
 - View Code: https://github.com/codekansas/gibberish-decoder
---

This is a simple experiment in predicting a word's meaning purely from it's spelling. A recurrent neural network is first trained to predict word embeddings from spelling, from a subset of 2000 word-embedding pairs [trained using Glove on a Wikipedia crawl](https://nlp.stanford.edu/projects/glove/). This subset is filtered for common words that are at least five letters long, so a lot of words won't appear in the search results. Next, a new word is fed to the network to get it's word embedding. The nearest neighbors to that embedding are shown; these represent the words that the network thinks are closest in meaning to the presented word. This happens entirely in your browser using <a href="https://github.com/transcranial/keras-js">Keras JS</a>. Nearest neighbor decoding is done using [NumJS](https://github.com/nicolaspanel/numjs).

<div class="ui segments">
    <div class="ui secondary header segment">
        Use a Neural Network to Infer Meaning from Gibberish
    </div>
    <div class="ui segment">
        <div class="ui fluid action icon input">
            <input type="text" placeholder="Enter Word Here" id="word">
            <button class="ui button" id="search-button">Search</button>
        </div>
    </div>
    <div class="ui segment" id="word-display">
        The nearest neighbor words will show up here.
    </div>
</div>

## Neural Networks in your Browser

Because the model is deployed using <a href="https://github.com/transcranial/keras-js">Keras JS</a>, you can run a full neural network in your browser, despite my site being hosted statically on [Github Pages](https://pages.github.com/) (although the network files are pretty large, so it can be kind of buggy, especially on slow connections). In the future I hope to make more interactive examples like this, for language modeling and other things.

{% include image.html description="The model architecture used to map words to their corresponding embeddings. This is a relatively simple recurrent neural network architecture. This model is trained on a subset of existing word embeddings." url="/resources/gibberish/encoder_network.png" %}

## Nearest Neighbor Decoding

[Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding) are a ubiquitous idea in language modeling that suggests that we can represent a word as a vector, where the location of the vector in vector space represents the meaning of that word. There are some [cool examples](https://www.quora.com/What-are-some-interesting-Word2Vec-results) of what this looks like in practice. The most common example is that if you take the vector for "king", subtract the vector for "man" and add the vector for "woman" you get the vector for "queen". This means that our neural network is essentially trying to predict the word's meaning purely from it's spelling. We can decode it's meaning by finding words with similar vectors. As with a lot of neural applications, interpreting the results is a bit like finding images in clouds.

{% include image.html description="Did you ever hear the tragedy of Darth Plagueis The Wise? If you type in \"plageus\", you end up with a vector embedding that is close (in vector space) to the vector embeddings for \"where\", \"opened\" and \"places\"." url="/resources/gibberish/vector_embeddings.png" %}

<script type="text/javascript" src="{{ "/resources/demos/keras.js" | prepend: site.baseurl }}" ></script>
<script type="text/javascript" src="{{ "/resources/demos/math.min.js" | prepend: site.baseurl }}" ></script>

<script>
function encode(inputString) {
    var arr = [];
    for (var i = 0; i < 30; i++) {
        if (i > inputString.length) {
            arr.push(0);
        } else {
            var v = inputString.charCodeAt(i);
            if (isNaN(v) || v < 97 || v > 122) {
                arr.push(0);
            } else {
                arr.push(v - 96);
            }
        }
    }
    return new Float32Array(arr);
}

// Writes words to the console.
function writeWords(toWhat, words) {
    var html = ['<div class="ui small header">Nearest Neighbors to "' + toWhat + '":</div>'];
    html.push('<div class="ui ordered list">');
    for (var i = 0; i < words.length; i++) {
        html.push('<div class="item">' + words[i] + '</div>');
    }
    html.push('</div>');
    $("#word-display").html(html.join(''));
}

// Loads the embeddings file.
function loadEmbeddings() {
    return new Promise(function (resolve, reject) {
        var xhr = new XMLHttpRequest();
        xhr.responseType = 'arraybuffer';
        xhr.open('GET', '{{ "/resources/gibberish/embeddings.buf" | prepend: site.baseurl }}', true);
        xhr.onload = function() {
            if (this.status >= 200 && this.status < 300) {
                resolve(new Float32Array(this.response));
            } else {
                reject({
                    status: this.status,
                    statusText: xhr.statusText
                });
            }
        }
        xhr.send();
    });
}
const embeddings = loadEmbeddings();

// Loads the words file.
function loadWords() {
    return new Promise(function (resolve, reject) {
        var xhr = new XMLHttpRequest();
        xhr.responseType = 'text';
        xhr.open('GET', '{{ "/resources/gibberish/words.txt" | prepend: site.baseurl }}', true);
        xhr.onload = function() {
            if (this.status >= 200 && this.status < 300) {
                resolve(this.response.split(','));
            } else {
                reject({
                    status: this.status,
                    statusText: xhr.statusText
                });
            }
        }
        xhr.send();
    });
}
const words = loadWords();

// Loads the model.
const model = new KerasJS.Model({
    filepaths: {
        model: '{{ "/resources/gibberish/model.json" | prepend: site.baseurl }}',
        weights: '{{ "/resources/gibberish/model_weights.buf" | prepend: site.baseurl }}',
        metadata: '{{ "/resources/gibberish/model_metadata.json" | prepend: site.baseurl }}'
    },
    gpu: true
});

var data;

// Sets up button actions once everything is loaded.
Promise.all([embeddings, model, words, $(document)]).then(values => {
    var embeddings = values[0], model = values[1], words = values[2];

    function process() {
        var word = $("#word").val();
        if (word) {
            const inputData = {
                'input': encode(word)
            }
            model.predict(inputData).then(result => {
                var vec = Array.prototype.slice.call(result.output), sims = [];

                // Calculates most similar.
                for (var i = 0; i < 2000; i++) {
                    var sarr = Array.prototype.slice.call(embeddings.subarray(i * 50, (i + 1) * 50));
                    var sim = math.multiply(vec, sarr);
                    sims.push(sim);
                }

                // Sorts words together in order of descending similarity.
                var list = [];
                for (var j = 0; j < sims.length; j++) {
                    list.push({'word': words[j], 'sim': sims[j]});
                }
                list.sort(function (a, b) {
                    return ((a.sim < b.sim) ? 1 : (a.sim == b.sim) ? 0 : -1);
                });

                // Gets most similar.
                var most_similar = [];
                for (var k = 0; k < 10; k++) {
                    most_similar.push(list[k].word);
                }

                writeWords(word, most_similar);
            });
        }
    }

    // Runs when the button is clicked.
    $("#search-button").click(process);

    // Runs when just pressing enter.
    $("#word").keypress(e => {
        if (e.which == 13) {
            process();
        }
    });
})
</script>
