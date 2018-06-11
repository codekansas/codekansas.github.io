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
    var html = ['<table class="table table-hover">'];
    html.push('<thead><tr><th>Neighbors</th><th>Score</th></tr></thead>');
    html.push('<tbody>');
    for (var i = 0; i < words.length; i++) {
        html.push('<tr><td>' + words[i].word + '</td><td>' + Math.floor(words[i].sim * 1000) / 1000 + '</td></tr>');
    }
    html.push('</tbody></table>');
    html.push('<div>Nearest neighbors to "' + toWhat + '"</div>');
    $("#word-display").html(html.join(''));
}

// Loads the embeddings file.
function loadEmbeddings() {
    return new Promise(function (resolve, reject) {
        var xhr = new XMLHttpRequest();
        xhr.responseType = 'arraybuffer';
        xhr.open('GET', '/assets/posts/gibberish/embeddings.buf', true);
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
        xhr.open('GET', '/assets/posts/gibberish/words.txt', true);
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
        model: '/assets/posts/gibberish/model.json',
        weights: '/assets/posts/gibberish/model_weights.buf',
        metadata: '/assets/posts/gibberish/model_metadata.json'
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

                writeWords(word, list.slice(0, 10));
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
