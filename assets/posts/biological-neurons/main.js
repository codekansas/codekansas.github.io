$(document).ready(() => {
  // Computes a good aspect ratio for the plots.
  function update_aspect_ratio() {
    const squeeze = $(window).width() < 800;
    const w = squeeze? 250 : 500, h = 100, r = (h / w) * 100;
    $(".neuron-plot")
      .attr("viewBox", "0 0 " + w + " " + h)
      .attr("preserveAspectRatio", "none");
    $(".limit-cycle-plot")
      .attr("viewBox", "0 0 " + (w / 2) + " " + h)
      .attr("preserveAspectRatio", "none");
    $("div.plot-container").css("padding-bottom", r + "%");
    return {"w": w, "h": h, "r": r};
  }
  function make_button() {
    return $("<button>")
      .attr("type", "button")
      .attr("class", "btn btn-default");
  }
  const plot_dims = update_aspect_ratio();

  function add_choices_controller(
    container,
    draw_func,
    model,
    configs,
  ) {
    for (i in configs) {
      const div = $("<div>")
        .attr("class", "btn-group")
        .attr("role", "group")
        .css("padding-bottom", "0.5em");
      const model_key = configs[i].key;
      for (j in configs[i].choices) {
        const choice = configs[i].choices[j];
        const choice_btn = make_button().html(choice);
        choice_btn.click(() => {
          model[model_key] = choice;
          choice_btn.addClass("active").siblings().removeClass("active");
          draw_func();
        });
        if (choice == configs[i].default) {
          choice_btn.addClass("active");
        }
        div.append(choice_btn);
      }
      container.append(div);
    }
  }

  // Adds "controllers" for various plots.
  function add_controller_from_funcs(
    ctrl_container,
    reset_func,
    val_func,
    dec_func,
    inc_func,
    draw_func,
  ) {
    // Creates the center button.
    const view_button = make_button().html(val_func());
    view_button.click(() => {
      reset_func();
      view_button.html(val_func());
      draw_func();
    });

    const make_direction_button = (direction, func) => {
      const span = $("<span>")
        .attr("class", "glyphicon glyphicon-chevron-" + direction);
      const button = make_button()
        .append(span)
        .click(() => {
          func();
          view_button.html(val_func());
          draw_func();
        });
      return button;
    }

    // Makes the accessory buttons..
    const left_button = make_direction_button("left", dec_func);
    const right_button = make_direction_button("right", inc_func);

    // Stacks the divs together in a group.
    const div = $("<div>")
      .attr("class", "btn-group")
      .attr("role", "group")
      .css("padding-bottom", "0.5em");
    div.append(left_button);
    div.append(view_button);
    div.append(right_button);
    ctrl_container.append(div);
  }

  // Short-hand function for adding plot controllers.
  function add_controller(
    key,
    def,
    label,
    unit,
    step_size,
    num_significant,
    draw,
    model,
    ctrl_container,
  ) {
    add_controller_from_funcs(
      ctrl_container,
      () => model[key] = def,
      () => label + ": " + (model[key] / unit).toFixed(num_significant),
      () => model[key] -= step_size * unit,
      () => model[key] += step_size * unit,
      draw,
    );
  }

  async function handle_lif_plot() {
    // Defines the default values.
    const v_reset = -60 * mV, v_thresh = 15 * mV, v_in = 20 * mV;
    const tau_m = 10 * ms, t_total = 100 * ms, dt = 1 * ms;

    // Initializes the model and plot area.
    const model = new LIFNeuron(tau_m, v_reset, v_thresh, v_in);
    const p = new Plot("#lif-neuron-plot", plot_dims.w, plot_dims.h, '#f2ae72');
    const draw = () => {
      model.reset();
      const voltages = model.run(t_total, dt);
      p.plot(voltages);
    }

    // Creates the element to store the controllers.
    const ctrl_container = $("<div>").attr("class", "btn-toolbar");
    $("#lif-neuron-plot").parent().after(ctrl_container);

    function add_controller_helper(key, def, name, unit, step, n_sig) {
      add_controller(key, def, name, unit, step, n_sig,
        draw, model, ctrl_container);
    }

    // Adds controller elements.
    add_controller_helper("v_reset", v_reset, "reset voltage (mV)", mV, 1, 0);
    add_controller_helper("v_thresh", v_thresh, "threshold voltage (mV)", mV, 0.5, 1);
    add_controller_helper("v_in", v_in, "input voltage (mV)", mV, 0.5, 1);
    add_controller_helper("tau_m", tau_m, "time constant (ms)", ms, 1, 0);

    // Redraws whenever the window is resized.
    draw();
  }

  async function handle_izhikevich_plot() {
    // Defines the default values.
    const a = 0.02, b = 0.2, c = -50, d = 2;
    const v_thresh = 30, v_in = 20, v_init = -60;
    const u_init = 0.5, tau_m = 10 * ms;
    const t_total = 100, dt = 0.01;

    // Initializes the model and plot area.
    const model = new IzhikevichNeuron(a, b, c, d, v_in,
        v_thresh, v_init, u_init);
    const p = new Plot("#izhikevich-neuron-plot", plot_dims.w, plot_dims.h,
      '#d96459');
    const draw = () => {
      model.reset();
      const voltages = model.run(t_total, dt);
      p.plot(voltages);
    }

    // Creates the element to store the controllers.
    const ctrl_container = $("<div>").attr("class", "btn-toolbar");
    $("#izhikevich-neuron-plot").parent().after(ctrl_container);

    function add_controller_helper(key, def, name, step, n_sig) {
      add_controller(key, def, name, 1, step, n_sig,
        draw, model, ctrl_container);
    }

    // Adds controller elements.
    add_controller_helper("a", a, "a", 0.001, 3);
    add_controller_helper("b", b, "b", 0.01, 2);
    add_controller_helper("c", c, "c", 1, 0);
    add_controller_helper("d", d, "d", 0.1, 1);
    add_controller_helper("v_in", v_in, "input voltage (mV)", 0.5, 1);
    add_controller_helper("v_thresh", v_thresh, "threshold voltage (mV)", 1, 0);
    add_controller_helper("v_init", v_init, "initial voltage (mV)", 1, 0);
    add_controller_helper("u_init", u_init, "initial u", 0.1, 1);

    // Redraws whenever the window is resized.
    draw();
  }

  async function handle_hodgkin_huxley_plot() {
    // Defines the default values.
    const e_na = 50.0, e_k = -77.0, e_l = -54.0;
    const g_na = 120.0, g_k = 36.0, g_l = 0.3;
    const v_in = 20.0, v_init = -65.0;
    const m_init = 0.05, h_init = 0.6, n_init = 0.32;
    const t_total = 100, dt = 0.01;
    const x_plot = 'v', y_plot = 'n';

    // Initializes the model and plot area.
    const model = new HodgkinHuxleyNeuron(e_na, e_k, e_l, g_na, g_k, g_l,
      v_in, v_init, m_init, h_init, n_init, x_plot, y_plot);
    const p = new Plot("#hodgkin-huxley-neuron-plot", plot_dims.w, plot_dims.h,
      '#8c4646');
    const l = new Plot("#hodgkin-huxley-limit-cycle", plot_dims.w / 2,
      plot_dims.h, '#52788b');
    const draw = () => {
      model.reset();
      const voltages = [], ns = [];
      for (let i = 0; i < t_total; i += dt) {
        model.step(dt);
        voltages.push([model.time, model.v]);
        ns.push([model[model.x_plot], model[model.y_plot]]);
      }
      p.plot(voltages);
      l.plot(ns);
    }

    // Creates the elements to control what limit cycles to plot.
    const limit_container = $("<div>").attr("class", "btn-toolbar");
    $("#hodgkin-huxley-limit-cycle").parent().after(limit_container);

    // Adds limit cycle container.
    add_choices_controller(
      limit_container,
      draw,
      model,
      [
        {
          'key': 'x_plot',
          'default': x_plot,
          'choices': ['v', 'n', 'm', 'h'],
        },
        {
          'key': 'y_plot',
          'default': y_plot,
          'choices': ['v', 'n', 'm', 'h'],
        }
      ]
    );

    // Creates the element to store the controllers.
    const ctrl_container = $("<div>").attr("class", "btn-toolbar");
    $("#hodgkin-huxley-neuron-plot").parent().after(ctrl_container);

    function add_controller_helper(key, def, name, step, n_sig) {
      add_controller(key, def, name, 1, step, n_sig,
        draw, model, ctrl_container);
    }

    // Adds controller elements.
    const cond_unit = "(mS / cm<sup>2</sup>)";
    add_controller_helper("e_na", e_na, "sodium reversal potential (mV)", 1, 0);
    add_controller_helper("e_k", e_k, "potassium reversal potential (mV)",
      1, 0);
    add_controller_helper("e_l", e_l, "leak potential (mV)", 1, 0);
    add_controller_helper("g_na", g_na, "sodium conductance " + cond_unit,
      1, 0);
    add_controller_helper("g_k", g_k, "potassium conductance " + cond_unit,
      0.5, 1);
    add_controller_helper("g_l", g_l, "leak conductance " + cond_unit,
      0.01, 2);
    add_controller_helper("v_in", v_in, "input voltage (mV)", 1, 0);
    add_controller_helper("v_init", v_init, "initial voltage (mV)", 1, 0);
    add_controller_helper("m_init", m_init, "initial m", 0.001, 3);
    add_controller_helper("h_init", h_init, "initial h", 0.01, 2);
    add_controller_helper("n_init", n_init, "initial n", 0.01, 2);

    // Redraws whenever the window is resized.
    draw();
  }

  // Retrieves the "draw" and "neuron" scripts.
  $.when(
    $.getScript("/assets/posts/biological-neurons/draw.js"),
    $.getScript("/assets/posts/biological-neurons/neuron.js"),
  ).done(() => {
    handle_lif_plot();
    handle_izhikevich_plot();
    handle_hodgkin_huxley_plot();
  });
});
