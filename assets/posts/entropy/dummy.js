$(document).ready(() => {
  const svg = d3.select("#dummy-circuit");

  // Defines the neuron width and height.
  const neuron_width = 20, neuron_height = 10;
  const e_color = "#f2ae72", i_color = "#8c4646", neuron_col = "#52788b";

  // Defines the neuron locations and colors.
  const neurons = [
    { "x": 60, "y": 20, "color" : neuron_col, "label": "A" },
    { "x": 60, "y": 80, "color" : neuron_col, "label": "B" },
    { "x": 140, "y": 20, "color" : neuron_col, "label": "C" },
    { "x": 140, "y": 80, "color" : neuron_col, "label": "D" },
    { "x": 30, "y": 50, "color": neuron_col, "label": "S" },
    { "x": 170, "y": 50, "color": neuron_col, "label": "O" },
  ];

  // Defines the connections between the neurons.
  const connections = [
    { "xa": 60, "xb": 140, "ya": 20, "yb": 20, "color": e_color, "width": 2 },
    { "xa": 60, "xb": 140, "ya": 80, "yb": 80, "color": e_color, "width": 2 },
    { "xa": 60, "xb": 140, "ya": 20, "yb": 80, "color": i_color, "width": 1 },
    { "xa": 60, "xb": 140, "ya": 80, "yb": 20, "color": i_color, "width": 1 },
    { "xa": 30, "xb": 60, "ya": 50, "yb": 20, "color": e_color, "width": 2 },
    { "xa": 30, "xb": 60, "ya": 50, "yb": 80, "color": e_color, "width": 2 },
    { "xa": 140, "xb": 170, "ya": 20, "yb": 50, "color": e_color, "width": 2 },
    { "xa": 140, "xb": 170, "ya": 80, "yb": 50, "color": e_color, "width": 2 },
  ]

  // Draws text onto the plot.
  const draw_text = async (svg, data, x_func, y_func) => {
    const text = svg
      .selectAll("text")
      .data(data)
      .enter()
      .append("text");

    const text_attributes = text
      .attr("x", x_func)
      .attr("y", y_func)
      .text(d => d.label)
      .attr("font-family", "monospace")
      .attr("font-size", "10px")
      .attr("fill", "white")
      .style("pointer-events", "none")
      .style("user-select", "none");
  }

  // Draws the connections between neurons.
  const draw_connections = async () => {
    const lines = svg.selectAll("line").data(connections).enter().append("line");
    const line_attributes = lines
      .style("stroke", d => d.color).style("stroke-width", d => d.width)
      .attr("x1", d => d.xa).attr("y1", d => d.ya)
      .attr("x2", d => d.xb).attr("y2", d => d.yb);
  }

  // Draws the neurons themselves.
  const draw_neurons = async () => {
    const circles = svg.selectAll("ellipse").data(neurons).enter()
      .append("ellipse");
    const circle_attributes = circles.attr("id", d => d.label)
      .attr("cx", d => d.x).attr("cy", d => d.y)
      .attr("rx", neuron_width).attr("ry", neuron_height)
      .style("fill", d => d.color);
    draw_text(svg, neurons, d => d.x - d.label.length * 3, d => d.y + 3);
  }

  // Pulses a neuron after a delay.
  const pulse = async (element, pulse_time) => {
    const pulse_part = pulse_time / 10;
    return element
      .transition()
      .duration(pulse_part)
      .attr("rx", neuron_width - 3)
      .attr("ry", neuron_height - 2)
      .transition()
      .duration(pulse_part * 3)
      .attr("rx", neuron_width + 10)
      .attr("ry", neuron_height + 5)
      .transition()
      .duration(pulse_part * 6)
      .attr("rx", neuron_width)
      .attr("ry", neuron_height);
  }

  const table = $("#dummy-table");
  const table_elems = ['s', 'a', 'b', 'c', 'd', 'o'].reduce((m, v) => {
    m[v] = {
      'count': { 'val': 0, 'elem': table.find('#' + v + '-count') },
      'entropy': { 'elem': table.find('#' + v + '-entropy') },
    };
    return m;
  }, {});

  const update = async (name, fired) => {
    const elem = table_elems[name];
    const count = fired ? ++elem.count.val : elem.count.val;
    const p = count / total;
    const ent = -(
      (p > 0 ? p * Math.log2(p) : 0) +
      (p < 1 ? (1 - p) * Math.log2(1 - p) : 0)
    );
    elem.count.elem.html(count);
    elem.entropy.elem.html(ent.toFixed(2));
  }

  // Defines the total number of stimulations.
  let total = 0;

  const ready_animations = async () => {
    const a = svg.select("#A"), b = svg.select("#B");
    const c = svg.select("#C"), d = svg.select("#D");
    const s = svg.select("#S"), o = svg.select("#O");

    // Sets up the pulse functions.
    let delay_interval = 100, pulse_time = 400;
    const pulse_o = async (ob, inc) => {
      if (ob) pulse(o, pulse_time);
      if (inc) update("o", ob);
    };
    const pulse_cd = async (cb, db, inc) => {
      if (cb) pulse(c, pulse_time);
      if (db) pulse(d, pulse_time);
      if (inc) {
        update("c", cb);
        update("d", db);
      }
      const ob = (cb || db);
      setTimeout(pulse_o, delay_interval, ob, inc);
    }
    const pulse_ab = async (ab, bb, inc) => {
      if (ab) pulse(a, pulse_time);
      if (bb) pulse(b, pulse_time);
      if (inc) {
        update("a", ab);
        update("b", bb);
      }
      const cb = ab && (bb ? Math.random() * 3 > 1 : true);
      const db = bb && (ab ? Math.random() * 3 > 1 : true);
      setTimeout(pulse_cd, delay_interval, cb, db, inc);
    }
    const pulse_s = async (inc) => {
      pulse(s, pulse_time);
      if (inc) update("s", true);
      const ab = Math.random() < 0.75, bb = Math.random() < 0.75;
      setTimeout(pulse_ab, delay_interval, ab, bb, inc);
    };

    // Adds the pulse listeners.
    s.on("click", () => pulse_s(false));
    a.on("click", () => pulse_ab(true, false, false));
    b.on("click", () => pulse_ab(false, true, false));
    c.on("click", () => pulse_cd(true, false, false));
    d.on("click", () => pulse_cd(false, true, false));
    o.on("click", () => pulse_o(true, false));

    // Sets a flag to terminate run.
    let break_flag = false;

    const reset = async () => {
      total = 0;
      break_flag = true;

      // Resets the table_elems data.
      Object.keys(table_elems).forEach(async (v) => {
        const elem = table_elems[v];
        elem.count.val = 0;
        elem.count.elem.html("0");
        elem.entropy.elem.html("1");
      });
    };

    const stimulate = async () => {
      total++;
      pulse_s(true);
    }

    // Adds reset handler.
    $("#dummy-reset").on("click", reset);
    $("#dummy-stim").on("click", stimulate);
    $("#dummy-run-100").on("click", () => {
      break_flag = false;
      const old_delay = delay_interval, old_pulse = pulse_time;
      delay_interval = 20;
      pulse_time = 100;
      (function repeat(i) {
        if (i == 0 || break_flag) {
          delay_interval = old_delay;
          pulse_time = old_pulse;
          return;
        }
        stimulate();
        setTimeout(repeat, delay_interval * 5, i - 1);
      })(100);
    });
    $("#dummy-stop-run").on("click", () => { break_flag = true; });
  }

  // Sets everything up.
  (async () => {
    draw_connections();
    await draw_neurons();
    ready_animations();
  })();
});
