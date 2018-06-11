$(document).ready(() => {
  let counts = null, words = {}, tfs = [];

  const make_head = function() {
    const cols = [].slice.call(arguments).map(x => '<th>' + x + '</th>');
    return $('<tr>' + cols + '</tr>');
  }

  const make_row = function() {
    const cols = [].slice.call(arguments).map(x => '<td>' + x + '</td>');
    return $('<tr>' + cols + '</tr>');
  }

  const filter_stops = words => words.reduce((a, x) => {
    if (STOPWORDS.has(x)) {
      return a;
    }
    a.push(x);
    return a;
  }, []);

  const update_counts = text => {
    // Hides everything if the text is empty.
    if (text.length == 0) {
      $('#new-doc-tokens-container').hide();
      $('#new-doc-stopped-container').hide();
      $('#new-doc-tf-container').hide();
      counts = null;
      return;
    }

    // Applies the porter stemmer.
    const stemmed = porter.stem(text);
    $('#new-doc-tokens-container').show();
    $('#new-doc-tokens').html(stemmed.join(' '));

    // Filters out stopwords.
    const stopped = filter_stops(stemmed);

    // Updates the "stopwords" field.
    if (stopped.length == 0) {
      $('#new-doc-stopped-container').hide();
      $('#new-doc-tf-container').hide();
      counts = null;
      return;
    }
    $('#new-doc-stopped-container').show();
    $('#new-doc-stopped').html(stopped.join(' '));

    // Counts the term frequency.
    counts = stopped.reduce((a, x) => {
      if (x.length == 0) return a;
      a[x] = a.hasOwnProperty(x) ? a[x] + 1 : 1;
      return a;
    }, {});
    $('#new-doc-tf-container').show();
    $('#new-doc-tf').html(Object.keys(counts)
      .map(x => '<b>' + x + '</b><sub>' + counts[x] + '</sub>').join(' '));
  }

  const TFIDF_HEAD = make_head('Term', 'Document Frequency', 'IDs');

  const update_tfidf = () => {
    const tb = $('#terms-body');
    tb.empty();
    tb.append(TFIDF_HEAD);
    for (let word of Object.keys(words)) {
      const doc_freq = words[word].length / tfs.length;
      tb.append(make_row(word, doc_freq.toFixed(3), words[word].join(', ')));
    }
  }

  const add_doc = doc => {
    doc.id = tfs.length;
    tfs.push(doc);

    // Adds the words to the word set.
    for (let word of Object.keys(doc.counts)) {
      if (words.hasOwnProperty(word)) {
        words[word].push(doc.id);
      } else {
        words[word] = [doc.id];
      }
    }
    update_tfidf();

    // Adds the document to the doc table.
    const count_text = Object.keys(doc.counts)
      .map(x => '<b>' + x + '</b><sub>' + doc.counts[x] + '</sub>').join(' ');
    const row = make_row(doc.id, doc.text, count_text);
    $('#docs-table').append(row);
  }

  const get_scores = (word, count) => {
    if (!words.hasOwnProperty(word)) {
      return null;
    }

    const df = words[word].length / tfs.length;
    const scores = words[word].map(docid => {
      const tf = tfs[docid].counts[word];
      const score = (1 + Math.log2(tf)) * Math.log2(1 + (1 / df));
      return { id: docid, tf: tf, score: score };
    });
    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, count);
  }

  const QUERY_HEAD = make_head(
    'ID', 'Term Frequency', 'Document Frequency', 'Score');

  $('#query-text').on('input', () => {
    const q = $('#query-text').val().replace(/\s/, '');
    $('#query-text').val(q);
    const stemmed = porter.stem_word(q);

    // Updates the table.
    const db = $('#queries-body');
    db.empty();
    db.append(QUERY_HEAD);

    const scores = get_scores(stemmed, 10);
    if (scores === null) return;

    const df = words[stemmed].length / tfs.length;
    for (let score of scores) {
      db.append(make_row(
        score.id, score.tf, df.toFixed(3), score.score.toFixed(3)));
    }
  });

  const QUERY_FULL_HEAD = make_head('ID', 'Text', 'Score');

  $('#query-full-text').on('input', () => {
    const stemmed = porter.stem($('#query-full-text').val());

    // Updates the table.
    const db = $('#queries-full-body');
    db.empty();
    db.append(QUERY_FULL_HEAD);

    const scores = stemmed
      .map(word => get_scores(word, 10))
      .reduce((a, s) => {
        if (s === null) return a;
        for (let x of s) {
          if (a.hasOwnProperty(x.id)) {
            a[x.id] += x.score;
          } else {
            a[x.id] = x.score;
          }
        }
        return a;
      }, {});

    // Sorts scores by IDs.
    const top_ids = Object.keys(scores)
      .sort((a, b) => scores[b].score - scores[a].score).slice(0, 10);

    for (let id of top_ids) {
      db.append(make_row(id, tfs[id].text, scores[id].toFixed(3)));
    }
  });

  const add_btn_click = () => {
    if (counts === null) return;

    const doc = {
      text: $('#new-doc-text').val(),
      counts: counts,
    };
    add_doc(doc);
    $('#new-doc-text').val('');
    update_counts('');
    $('#new-doc-text').focus();
  }

  $('#new-doc-add-btn').click(add_btn_click);

  $('#new-doc-text').on('input', function() {
    update_counts($(this).val());
  });

  $('#new-doc-text').keyup(function(e){
    if(e.keyCode == 13) {
      add_btn_click();
    }
  });
});
