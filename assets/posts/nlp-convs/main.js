const BASE_PATH = '/assets/posts/nlp-convs/';
const get_path = fname => BASE_PATH + fname;

const FILTER_RE = new RegExp(
  '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    .split('')
    .map(i => '\\' + i)
    .join('|'),
  'g');

const PRED_WAIT_TIME = 500;

const get_examples = async () => {
  const response = await fetch(get_path('examples.json'));
  const data = await response.json();
  return data;
}

const get_word_to_index_map = async () => {
  const response = await fetch(get_path('word_to_index.json'));
  const data = await response.json();
  const wordMap = new Map();
  for (let k of Object.keys(data)) {
    wordMap.set(k, data[k]);
  }
  return wordMap;
};

const get_knns = async () => {
  const response = await fetch(get_path('knns.json'));
  const data = await response.json();
  return data;
}

const text_to_word_sequence = text => text
  .toLowerCase()
  .replace(FILTER_RE, ' ')
  .split(/ +/)
  .filter(word => word.length > 0);

const scale_red_to_green = val => {
  const r = Math.floor(Math.max(0, 1 - (1.5 * val)) * 2);
  const g = Math.floor(Math.max(0, val * 1.5 - 0.5) * 2);
  const str = 'rgb(' + r + ', ' + g + ', 0)';
  return str;
}

$('#sentiment-test-output').hide();

Promise.all([
  tf.loadModel(get_path('model/model.json')),
  get_word_to_index_map(),
  get_examples(),
]).then(values => {
  const model = values[0],
        word_to_index = values[1],
        examples = values[2];
  const sentiment_test_output = d3.select('#sentiment-test-output');

  const get_sentence_predictions = async word_seq => {
    const SMOOTH_LENGTH = 3;
    const idxs = word_seq
      .map(i => (word_to_index.get(i) || -1) + 3);
    idxs.unshift(1);
    const tensor = tf.tensor2d(idxs, [1, idxs.length]);
    const predictions = model.predict(tensor).slice([0, 1, 0]);
    const preds_smooth = tf.div(tf.conv1d(
      predictions,
      tf.ones([SMOOTH_LENGTH, 1, 1]),
      1,
      'same',
    ), SMOOTH_LENGTH);
    return preds_smooth;
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

    $('[data-toggle="popover"]').popover()
  };

  let pred_interval = null;
  $('#sentiment-test-input').on('input', function(e) {
    clearTimeout(pred_interval);
    const sentence = $(this).val();
    pred_interval = setTimeout(
      () => update_sentiment_output(sentence),
      PRED_WAIT_TIME);
  });

  d3.select('#sentiment-test-input-group')
    .insert('div', ':first-child')
    .style('padding', '0 1em 1em 1em')
    .selectAll('button')
    .data(examples)
    .enter()
    .append('button')
    .attr('type', 'button')
    .attr('class', 'btn btn-info')
    .style('margin', '0.2em')
    .text((d, i) => 'Example ' + (i + 1))
    .on('click', d => {
      $('#sentiment-test-input').val(d);
      update_sentiment_output(d);
    });
});

Promise.all([
  get_knns(),
]).then(values => {
  const knns = values[0];
  const knn_vis = d3.select('#convolution-knn-vis');

  const update_func = (d, i) => {
    knn_vis.select('div').remove();

    const div = knn_vis.append('div');

    div.append('h3')
      .text('Layer ' + (i + 1) + ' Receptive Field')

    const sub_divs = div.selectAll('div')
      .data(d)
      .enter()
      .append('div')
      .style('padding-bottom', '1em');

    sub_divs.append('h4')
      .text((_, i) => 'Position ' + (i + 1))

    sub_divs.selectAll('span')
      .data(i => i.words)
      .enter()
      .append('span')
      .attr('class', 'label label-primary')
      .text(i => i);
  }

  d3.select('#convolution-knns')
    .append('div')
    .selectAll('button')
    .data(knns)
    .enter()
    .append('button')
    .style('margin', '0.2em')
    .style('width', '3em')
    .attr('class', 'btn btn-primary btn-xs')
    .text((_, i) => (i + 1))
    .on('click', update_func);
});
