// Defines the index-word and word-index lookup dicts.
const index_to_token = {
  0: 'n', 1: ',', 2: ' ', 3: 'm', 4: 'o', 5: 'c', 6: 'S', 7: 'Q', 8: 'J',
  9: 'v', 10: 'x', 11: 'K', 12: 'N', 13: 'R', 14: '!', 15: ';', 16: ']',
  17: ' ', 18: 'U', 19: 'z', 20: 'I', 21: 'j', 22: 'T', 23: 'E', 24: '&',
  25: 'F', 26: 'd', 27: 'H', 28: 'O', 29: 's', 30: 'f', 31: 'A', 32: ':',
  33: 'X', 34: 'G', 35: '\t', 36: 'r', 37: 'y', 38: 'Y', 39: 'l', 40: 'q',
  41: 'P', 42: 'V', 43: 'k', 44: 'D', 45: 'B', 46: 'i', 47: 'u', 48: 'e',
  49: 'h', 50: 'p', 51: '?', 52: '.', 53: 't', 54: 'g', 55: 'C', 56: '[',
  57: 'W', 58: 'Z', 59: 'w', 60: 'b', 61: 'M', 62: "'", 63: 'L', 64: 'a',
  65: '-',
}

// Defines the reverse mappings.
const token_to_index = Object.keys(index_to_token).reduce(
  (a, k) => {
    a[index_to_token[k]] = parseInt(k);
    return a;
  }, {}
);

const model = new KerasJS.Model({
  filepaths: {
    model: '/assets/posts/shakespeare/model.json',
    weights: '/assets/posts/shakespeare/weights.buf',
    metadata: '/assets/posts/shakespeare/metadata.json'
  },
  gpu: true,  // Force use the WebGL binaries (otherwise breaks).
});

Promise.all([model, $(document)]).then(values => {
  const model = values[0];
  const button_choices = $("#button-choices");
  const output_text = $("#output-text");

  // Starts off with all the keys having equal sampling probability.
  var indices = Object.keys(index_to_token).map((k) => [1, k]);

  // Resets the GRU states.
  async function reset_states() {
    const layer_map = await model.modelLayersMap;
    ['gru_16', 'gru_17', 'gru_18'].forEach(layer_name => {
      const layer = layer_map.get(layer_name);
      layer.currentHiddenState = new KerasJS.Tensor([], [layer.units]);
    });
  }

  async function add_button_choices(indices) {
    button_choices.empty();
    indices.forEach((index) => {
      const new_button = $("<button>", {
        "type": "button",
        "class": "btn btn-default",
        "style": "width: 3em;",
        "role": "group",
      });
      new_button.html(index_to_token[index].replace(' ', '_'));
      new_button.click(() => {
        step(index);
      });
      button_choices.append(new_button);
    });
  }

  async function sort_indices(predictions) {
    var pred_indices = [];
    for (var i = 0; i < predictions.length; i++) {
      pred_indices.push([predictions[i], i]);
    }
    pred_indices.sort((a, b) => b[0] - a[0]);
    return pred_indices;
  }

  async function step(index) {
    if (!index_to_token.hasOwnProperty(index)) {
      return;
    }

    // Appends the token to the output display.
    output_text.html((_, c) => c + index_to_token[index]);

    // Gets the predictions from the model.
    const prediction = await model.predict({
      'input_sequence': new Float32Array([index]),
    }).then(result => { return result.prediction; });

    // Parses the predicted indices.
    indices = await sort_indices(prediction);

    // Updates the bar chart.
    const total = indices.reduce((a, i) => a + i[0], 0);
    d3.select(".chart")
      .selectAll("div")
      .data(indices)
      .style("width", (d) => { return (3 + (d[0] / total) * 97) + "%"; })
      .text(function(d) { return index_to_token[d[1]].replace(' ', '_'); });

    // Gets the top 5 indices.
    var choices = [];
    for (var i = 0; i < 5; i++) {
      choices.push(indices[i][1]);
    }
    return add_button_choices(choices);
  }

  async function random_choice() {
    const choice_val = math.random(indices.reduce((a, i) => a + i[0], 0));
    var v = 0, choice = 0;
    for (var i = 0, len = indices.length; i < len; i++) {
      v += indices[i][0];
      if (v > choice_val) {
        choice = indices[i][1];
        break;
      }
    }
    step(choice);
  }

  async function start() {
    output_text.empty();
    const tokens = ['A', 'd', 'v', 'N', 'e', 't'];
    const choices = tokens.map((i) => token_to_index[i]);
    add_button_choices(choices);
  }

  // Activates the reset button.
  $("#reset-button").click(() => {
    reset_states();
    start();
  });

  // Activates the random choice button.
  $("#random-button").click(() => {
    random_choice();
  });

  // Adds keypress option.
  $(document).on("keypress", (e) => {
    $("#reset-button").blur();
    $("#random-button").blur();

    if (e.which == 13) {  // Enter
      random_choice();
      return;
    }

    const c = String.fromCharCode(e.which);
    if (token_to_index.hasOwnProperty(c)) {
      step(token_to_index[c]);
    }

    return false;
  });

  start();
});
