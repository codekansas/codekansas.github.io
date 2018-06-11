const filter_stops = words => words.reduce((a, x) => {
  if (STOPWORDS.has(x)) {
    return a;
  }
  a.push(x);
  return a;
}, []);

const update_idxs = (idxs, idx, text) => {
  return porter.stem(text).reduce((a, x) => {
    if (x.length == 0 || STOPWORDS.has(x)) return a;
    if (!a.hasOwnProperty(x)) {
      a[x] = {};
    }

    if (a[x].hasOwnProperty(idx)) {
      a[x][idx]++;
    } else {
      a[x][idx] = 1;
    }
    return a;
  }, idxs);
};

// Gets document indices.
let idxs = {};
for (let i = 0; i < DATA.length; i++) {
  idxs = update_idxs(idxs, i, DATA[i].text);
}

// Divides by document frequency.
for (let k of Object.keys(idxs)) {
  const df = Math.log2(1 + (DATA.length / Object.keys(idxs[k]).length));
  for (let j of Object.keys(idxs[k])) {
    const tf = idxs[k][j];
    idxs[k][j] = (1 + Math.log2(tf)) * df;
  }
}

const make_head = function() {
  const cols = [].slice.call(arguments).map(x => '<th>' + x + '</th>');
  return $('<tr>' + cols + '</tr>');
}

const make_row = function() {
  const cols = [].slice.call(arguments).map(x => '<td>' + x + '</td>');
  return $('<tr>' + cols + '</tr>');
}

// Puts <b> and </b> around the term.
const get_indices = (toks, stemmed, term) => {
  let fixed = [];
  for (let i = 0; i < stemmed.length; i++) {
    if (stemmed[i] == term) {
      fixed.push('<b>');
      fixed.push(toks[i]);
      fixed.push('</b>');
    } else {
      fixed.push(toks[i]);
    }
  }
  return ids;
};

// Makes the indices with each term bold.
const make_bold = (text, terms) => {
  const toks = text.split(/([^a-zA-Z']+)/);
  const stemmed = porter.stem(toks.map(x => x.toLowerCase()));
  const tset = new Set(terms);

  // Converts tokens to highlighted.
  for (let i = 0; i < stemmed.length; i++) {
    if (tset.has(stemmed[i])) {
      toks[i] = '<span style="background-color: #FFFF00">' + toks[i] + '</span>';
    }
  }

  // Appends "fixed" tokens.
  let fixed = [];
  for (let i = 0; i < stemmed.length; i++) {
    if (tset.has(stemmed[i])) {
      fixed.push(toks.slice(Math.max(i - 10, 0), i + 10).join(''));
      i += 10;
    }
  }

  return fixed.join(' ... ');
};

$(document).ready(function() {
  const QUERY_FULL_HEAD = make_head('Text', 'Page');
  const db = $('#queries-full-body');
  const cbk = $('#query-all');

  const update_table = () => {
    const stemmed = filter_stops(porter.stem($('#query-full-text').val()));

    // Gets the summed counts of each term.
    const counts = stemmed.reduce((a, x) => {
      if (!idxs.hasOwnProperty(x)) return a;
      return Object.keys(idxs[x]).reduce((b, y) => {
        const score = idxs[x][y];
        b[y] = b.hasOwnProperty(y) ? b[y] + score : score;
        return b;
      }, a);
    }, {});

    // Sorts by number of page appearances.
    const sorted = Object.keys(counts).sort((a, b) => {
      if (counts[a] == counts[b]) {
        return DATA[a].page - DATA[b].page;  // Sort ascending pages.
      } else {
        return counts[b] - counts[a];  // Sort descending frequency.
      }
    });

    // Adds columns to the table.
    db.empty();
    db.append(QUERY_FULL_HEAD);

    const maxv = cbk.prop('checked') ? sorted.length : 5;
    for (let i = 0; i < Math.min(maxv, sorted.length); i++) {
      const d = DATA[sorted[i]];
      const t = make_bold(d.text, stemmed);
      db.append(make_row(t, d.page));
    }

    // Adds extra parts.
    if (sorted.length > maxv) {
      const pages = 'Also on ' + sorted.slice(5).map(x => {
        return 'pg. ' + DATA[x].page;
      }).join(', ');
      db.append(make_row(pages, ''));
    }
  };

  update_table();
  $('#query-full-text').on('input', update_table);
  cbk.change(update_table);
});
