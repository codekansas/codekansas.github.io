const BASE_PATH = '/assets/posts/nlp_convs/tfjs_model/';
const get_path = fname => BASE_PATH + fname;

const FILTER_RE = new RegExp(
  '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    .split('')
    .map(i => '\\' + i)
    .join('|'),
  'g');

const PRED_WAIT_TIME = 500;

const get_word_to_index_map = async () => {
  const response = await fetch(get_path('word_to_index.json'));
  const data = await response.json();
  const wordMap = new Map();
  for (let k of Object.keys(data)) {
    wordMap.set(k, data[k]);
  }
  return wordMap;
};

const text_to_word_sequence = text => text
  .toLowerCase()
  .replace(FILTER_RE, ' ')
  .split(/ +/)
  .filter(word => word.length > 0);

const scale_red_to_green = val => {
  const r = Math.floor(Math.max(0, 1 - (1.5 * val)) * 255);
  const g = Math.floor(Math.max(0, val * 1.5 - 0.5) * 255);
  const str = 'rgb(' + r + ', ' + g + ', 0)';
  return str;
}

$('#sentiment-test-output').hide();

Promise.all([
  tf.loadModel(get_path('model.json')),
  get_word_to_index_map(),
]).then(values => {
  const model = values[0], word_to_index = values[1];
  const sentiment_test_output = d3.select('#sentiment-test-output');

  const get_sentence_predictions = async word_seq => {
    const idxs = word_seq
      .map(i => (word_to_index.get(i) || -1) + 3);
    idxs.unshift(1);
    const tensor = tf.tensor2d(idxs, [1, idxs.length]);
    const predictions = model.predict(tensor).as1D().slice(1);
    return predictions;
  };

  const update_sentiment_output = async sentence => {
    const word_seq = text_to_word_sequence(sentence);
    if (word_seq.length == 0) {
      $('#sentiment-test-output').hide();
      return;
    } else {
      $('#sentiment-test-output').show();
    }
    const preds = await get_sentence_predictions(word_seq);
    const word_preds = Array.from(preds.dataSync()).map((p, i) => ({
      word: word_seq[i],
      pred: p,
    }));

    sentiment_test_output.select('div').remove();

    sentiment_test_output
      .append('div')
      .selectAll('span')
      .data(word_preds)
      .enter()
      .append('span')
      .attr('class', 'label')
      .text(d => d.word)
      .style('background-color', d => scale_red_to_green(d.pred));
  };

  let pred_interval = null;
  $('#sentiment-test-input').on('input', function(e) {
    clearTimeout(pred_interval);
    const sentence = $(this).val();
    pred_interval = setTimeout(
      () => update_sentiment_output(sentence),
      PRED_WAIT_TIME);
  });
});
