---
layout: post
title: Biological Neuron Models
date: 2017-09-20 12:00:00
categories: neuroscience visualizations
excerpt: >
  An exploration of some biological neuron models, in Javascript
---

<style>
div.plot-container {
  position: relative;
  width: 100%;
  margin-top: 1em;
  margin-bottom: 1em;
}
svg.neuron-plot {
  position: absolute;
  width: 100%;
  height: 100%;
  border: 1px solid grey;
}
svg.limit-cycle-plot {
  position: absolute;
  width: 50%;
  height: 100%;
  border: 1px solid grey;
  align: center;
}
</style>

<script defer src="{{ site.cdn.d3js }}"></script>
<script defer src="/assets/posts/biological-neurons/main.js"></script>

## Introduction

- Different types of biological neuron models are presented below, with controls for the various parameters they use
- The models are arranged (roughly) in order from least detailed to most detailed
- The vertical axis is the voltage of the neuron in millivolts, and the horizontal axis is the current time in milliseconds
- For each model, the initial input voltage is zero, and stepped to the specified voltage after 20 milliseconds

## Leaky Integrate-and-Fire

<div class="plot-container">
  <svg class="neuron-plot no-select" id="lif-neuron-plot"></svg>
</div>

### Notes

- This is the simplest neuron model out there
- Considers the neuron's membrane to be an RC circuit, hence the characteristic RC curve
- Each drop represents a spike; because the circuit is so stereotyped, spikes can be predicted exactly given knowledge of the neuron's input voltages

## Izhikevich

<div class="plot-container">
  <svg class="neuron-plot no-select" id="izhikevich-neuron-plot"></svg>
</div>

### Notes

- This model is able to capture a wide variety of firing dynamics with an exceptionally small number of parameters
- Check out [this website](https://www.izhikevich.org/publications/spikes.htm) for some parameters that do cool stuff

## Hodgkin-Huxley

<div class="plot-container">
  <svg class="neuron-plot no-select" id="hodgkin-huxley-neuron-plot"></svg>
</div>

### Notes

- This model is based on detailed patch clamp experiments, and models spiking at an ionic level
- This model was created in 1952

<div class="plot-container">
  <svg class="limit-cycle-plot" id="hodgkin-huxley-limit-cycle"></svg>
</div>

### Limit Cycles

- The plot above compares two of the Hogdkin Huxley model's gating variables at various times
- The first choice is the variable that is displayed on the X axis, and the second choice is the variable this is displayed on the Y axis
