// Defines some standard units.
const ms = 0.001, mV = 0.001, mS = 0.001, cm = 0.01;


class NeuronModel {
  constructor() {
    if (new.target === NeuronModel) {
      throw new TypeError("Cannot instantiate NeuronModel directly");
    }
    this.reset();
  }

  reset() {
    this.time = 0;
    this.v_mem = 0;
  }

  step(dt) {
    this.time += dt;
  }

  run(time_len, dt) {
    const r = [];
    for (let i = 0; i < time_len; i += dt) {
      this.step(dt);
      r.push([this.time, this.v_mem / mV]);
    }
    return r;
  }
}


class LIFNeuron extends NeuronModel {
  constructor(tau_m, v_reset, v_thresh, v_in) {
    super();
    this.tau_m = tau_m;
    this.v_reset = v_reset;
    this.v_thresh = v_thresh;
    this.v_in = v_in;
  }

  step(dt) {
    super.step(dt / ms);
    const v_in = this.time > 20 ? this.v_in : this.v_reset;
    const dv = ((v_in - this.v_mem) / this.tau_m) * dt;
    this.v_mem += dv;
    this.v_mem = this.v_mem > this.v_thresh ? this.v_reset : this.v_mem;
  }
}


class IzhikevichNeuron extends NeuronModel {
  constructor(a, b, c, d, v_in, v_thresh, v_init, u_init) {
    super();
    this.a = a;
    this.b = b;
    this.c = c;
    this.d = d;
    this.v_in = v_in;
    this.v_thresh = v_thresh;
    this.v_init = v_init;
    this.u_init = u_init;
    this.reset();
  }

  reset() {
    super.reset();
    this.u = this.u_init;
    this.v = this.v_init;
    this.v_mem = this.v * mV;
  }

  step(dt) {
    super.step(dt);

    // Starts input voltage after 20 milliseconds.
    const v_in = this.time > 20 ? this.v_in : 0;

    // Computes the updates.
    const dv = (
      0.04 * this.v * this.v +
      5 * this.v + 140 -
      this.u + v_in
    ) * dt;
    const du = (this.a * (this.b * this.v - this.u)) * dt;

    // Adds the updates.
    this.v += dv;
    this.u += du;

    // Checks the threshold voltage.
    if (this.v >= this.v_thresh) {
      this.v = this.c;
      this.u += this.d;
    }

    this.v_mem = this.v * mV;
  }
}


class HodgkinHuxleyNeuron extends NeuronModel {
  constructor(
    e_na,
    e_k,
    e_l,
    g_na,
    g_k,
    g_l,
    v_in,
    v_init,
    m_init,
    h_init,
    n_init,
    x_plot,
    y_plot,
  ) {
    super();

    // Defines some of the model's constants.
    this.e_na = e_na;
    this.e_k = e_k;
    this.e_l = e_l;
    this.g_na = g_na;
    this.g_k = g_k;
    this.g_l = g_l;
    this.v_in = v_in;

    // Defines the model variables.
    this.v_init = v_init;
    this.m_init = m_init;
    this.h_init = h_init;
    this.n_init = n_init;

    // For plotting limit cycles.
    this.x_plot = x_plot;
    this.y_plot = y_plot;

    this.reset();
  }

  reset() {
    super.reset();
    this.v = this.v_init;
    this.m = this.m_init;
    this.h = this.h_init;
    this.n = this.n_init;
    this.v_mem = this.v * mV;
  }

  // Defines some functions to compute the ion channel characteristic updates.
  alpha_m(v) {
    const v_offset = v + 40.0;
    return 0.1 * v_offset / (1.0 - Math.exp(-v_offset / 10.0));
  }
  alpha_h(v) { return 0.07 * Math.exp(-(v + 65.0) / 20.0); }
  alpha_n(v) {
    const v_offset = v + 55.0;
    return 0.01 * v_offset / (1.0 - Math.exp(-v_offset / 10.0));
  }
  beta_m(v) { return 4.0 * Math.exp(-(v + 65.0) / 18.0); }
  beta_h(v) { return 1.0 / (1.0 + Math.exp(-(v + 35.0) / 10.0)); }
  beta_n(v) { return 0.125 * Math.exp(-(v + 65.0) / 80.0); }

  // Defines some functions to compute the various currents.
  i_na(v, m, h) { return this.g_na * m * m * m * h * (v - this.e_na); }
  i_k(v, n) {
    const n_half = n * n;  // 2 ops instead of 3, got`em.
    return this.g_k * n_half * n_half * (v - this.e_k);
  }
  i_l(v) { return this.g_l * (v - this.e_l); }

  step(dt) {
    super.step(dt);

    // Starts input voltage after 20 milliseconds.
    const v_in = this.time > 20 ? this.v_in : 0;

    // Computes the updates.
    const dv = (
      v_in -
      this.i_na(this.v, this.m, this.h) -
      this.i_k(this.v, this.n) -
      this.i_l(this.v, this.n)
    ) * dt;
    const dm = (
      this.alpha_m(this.v) * (1.0 - this.m) -
      this.beta_m(this.v) * this.m
    ) * dt;
    const dh = (
      this.alpha_h(this.v) * (1.0 - this.h) -
      this.beta_h(this.v) * this.h
    ) * dt;
    const dn = (
      this.alpha_n(this.v) * (1.0 - this.n) -
      this.beta_n(this.v) * this.n
    ) * dt;

    // Adds the updates.
    this.m += dm;
    this.h += dh;
    this.n += dn;
    this.v += dv;

    // Updates the voltages.
    this.v_mem = this.v * mV;
  }
}
