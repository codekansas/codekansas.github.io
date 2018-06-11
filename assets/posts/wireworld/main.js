const CONDUCTOR = 0, HEAD = 1, TAIL = 2;
const NODE_RADIUS = 3;
const EVENT_TIME = 100;  // ms
const N_WIDE = 10, N_TALL = 10;

// Maps nodes to their respective colors.
const get_color = value => {
  switch (value) {
    case HEAD:        return '#8c4646';
    case TAIL:        return '#d96459';
    case CONDUCTOR:
    default:          return '#f2ae72';
  }
}

class Node {
  constructor(x, y) {
    this.state = CONDUCTOR;
    this.connections = new Set();
    this.next_state = CONDUCTOR;
    this.x = x;
    this.y = y;
    this.g = null;
  }

  dist(node) {
    const dx = node.x - this.x, dy = node.y - this.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  async draw() {
    const x = this.x, y = this.y;

    // Clears the elements.
    this.g.selectAll('line').remove();

    return this.g
      .selectAll('line')
      .data(Array.from(this.connections))
      .enter()
      .append('line')
      .each(function(d) {
        let dx = d.x - x, dy = d.y - y;
        const ratio = NODE_RADIUS / Math.sqrt(dx * dx + dy * dy);
        dx *= ratio;
        dy *= ratio;

        d3.select(this)
          .attr('x1', d.x - dx)
          .attr('y1', d.y - dy)
          .attr('x2', x + dx)
          .attr('y2', y + dy)
          .attr('marker-end', 'url(#triangle)')
          .attr('class', 'connections');
      });
  }

  toggle_connection(node) {
    if (this.connections.has(node)) {
      this.connections.delete(node);
    } else {
      this.connections.add(node);
    }
  }

  // Defines the logic for updating the next state.
  async step() {
    switch (this.state) {
      case HEAD: this.next_state = TAIL; break;
      case TAIL: this.next_state = CONDUCTOR; break;
      case CONDUCTOR:
      default:
        let num_heads = 0;
        for (let node of this.connections) {
          if (node.state == HEAD) {
            num_heads++;
          }
        }
        switch (num_heads) {
          case 1:
          case 2: this.next_state = HEAD; break;
          default: this.next_state = CONDUCTOR; break;
        }
    }
  }
}

// Contains the current selected node.
let selected_node = null;

// Holds all the nodes on the screen.
const nodes = [];

// Handles when a cell is clicked.
const cell_on_click = async (d, i, c, svg) => {
  if (selected_node == null) {
    selected_node = { 'cell': c[i], 'node': d };
    d3.select(selected_node.cell).classed('cell-selected', true);
  } else {
    if (selected_node.cell == c[i]) {
      d.state = (d.state + 1) % 3;
      d3.select(c[i]).style('fill', get_color(d.state));
    } else {
      d.toggle_connection(selected_node.node);
      d.draw();
    }

    // Un-selects the selected node.
    d3.select(selected_node.cell).classed('cell-selected', false);
    selected_node = null;
  }
}

let restart_on_none = true;

const event_loop = async color_func => {
  await nodes.forEach(async n => n.step());

  // Updates states, checking if any of them change.
  const state_change = await nodes.reduce(async (a, n) => {
    if (n.next_state != n.state) {
      a = true;
      n.state = n.next_state;
    }
    return a;
  }, false);

  // Randomly sparks a node.
  if (restart_on_none && !state_change) {
    const node = nodes[Math.floor(Math.random() * nodes.length)];
    node.state = HEAD;
    await node.draw();
  }

  await color_func();
  setTimeout(event_loop, EVENT_TIME, color_func);
}

$(document).ready(() => {
  const svg = d3.select('#automaton');

  // Adds all the nodes.
  const node_lattice = [];
  for (let j = 0; j < N_TALL; j++) {
    const node_row = [];
    for (let i = 0; i < N_WIDE; i++) {
      const node = new Node(
        i * (100 / N_WIDE) + (50 / N_WIDE),
        j * (100 / N_TALL) + (50 / N_TALL),
      );
      nodes.push(node);
      node_row.push({ 'node': node, 'visited': false });
    }
    node_lattice.push(node_row);
  }
  nodes.forEach(n => { n.g = svg.append('g'); });

  // Draws the circles themselves.
  const circles = svg
    .selectAll('circle')
    .data(Array.from(nodes))
    .enter()
    .append('circle')
    .attr('cx', d => d.x)
    .attr('cy', d => d.y)
    .attr('r', NODE_RADIUS)
    .on('click', (d, i, c) => cell_on_click(d, i, c, svg));

  // Adds "g" elements to the nodes.
  const color_func = () => circles.style('fill', d => get_color(d.state));

  const shuffle = a => {
      for (let i = a.length; i > 0; i--) {
          const j = Math.floor(Math.random() * i);
          const x = a[i - 1];
          a[i - 1] = a[j];
          a[j] = x;
      }
      return a;
  }

  // Contains the possible directions to go.
  const possible = [
    { 'i': 1, 'j': 0 }, { 'i': -1, 'j': 0 },
    { 'i': 1, 'j': 1 }, { 'i': -1, 'j': 1 },
    { 'i': 1, 'j': -1 }, { 'i': -1, 'j': -1 },
    { 'i': 0, 'j': 1 }, { 'i': 0, 'j': -1 },
    // { 'i': 2, 'j': 1 }, { 'i': 1, 'j': 2 },
    // { 'i': -2, 'j': 1 }, { 'i': -1, 'j': 2 },
    // { 'i': 2, 'j': -1 }, { 'i': 1, 'j': -2 },
    // { 'i': -2, 'j': -1 }, { 'i': -1, 'j': -2 },
  ];

  const clear_connections = async () => {
    nodes.forEach(async n => { n.connections.clear(); n.draw(); });
  }

  // DFS to connect all the nodes together.
  const randomize_connections = async use_dfs => {
    await clear_connections();
    node_lattice.forEach(n => n.forEach(i => { i.visited = false; }));
    const node_queue = [], init_node = {
      'i': Math.floor(Math.random() * N_WIDE),
      'j': Math.floor(Math.random() * N_TALL),
    };
    node_queue.push(init_node);
    node_lattice[init_node.j][init_node.i].visited = true;
    while (node_queue.length > 0) {
      const idx = (use_dfs ? node_queue.pop() : node_queue.shift());
      const node = node_lattice[idx.j][idx.i];
      for (let next of shuffle(possible)) {
        const ni = idx.i + next.i, nj = idx.j + next.j;
        if (ni >= 0 && ni < N_WIDE && nj >= 0 && nj < N_TALL) {
          if (!node_lattice[nj][ni].visited) {
            node_lattice[nj][ni].visited = true;
            node_lattice[nj][ni].node.toggle_connection(node.node);
            node.node.toggle_connection(node_lattice[nj][ni].node);
            node_queue.push({ 'i': ni, 'j': nj });
          }
        }
      }
    }
    nodes.forEach(n => n.draw());
  }

  // Adds buttton callbacks.
  $('#clear-connections').click(clear_connections);
  $('#randomize-connections').click(async () => randomize_connections(true));
  $('#randomize-bfs').click(async () => randomize_connections(false));
  $('#toggle-sparking').click(async () => {
    restart_on_none = !restart_on_none;
  });

  // Starts the event loop.
  const start = async () => {
    await randomize_connections(true);
    event_loop(color_func);
  };
  start();
});
