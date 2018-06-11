---
layout: post
title: Generating Shakespeare
date: 2017-09-18 12:00:00
categories: deep-learning visualizations
excerpt: >
  A visualization of generating text using an RNN
---

<script defer src="{{ site.cdn.d3js }}"></script>
<script defer src="{{ site.cdn.mathjs }}"></script>
<script defer src="{{ site.cdn.kerasjs }}"></script>
<script defer src="/assets/posts/shakespeare/sequence.js"></script>

<h2 id="output-text"></h2>

<style>
.chart {
  border: 1px solid black;
  padding-right: 1px;
  padding-left: 1px;
}
.chart, .choice-bar {
  margin-top: 1em;
}
.chart div {
  font: 10px sans-serif;
  text-align: right;
  padding: 3px;
  margin-top: 1px;
  margin-bottom: 1px;
  color: white;
  width: 100%;
}
</style>

<div class="chart" width="100%">
  <div class="bgcolor-a">A</div>
  <div class="bgcolor-b">d</div>
  <div class="bgcolor-c">v</div>
  <div class="bgcolor-b">N</div>
  <div class="bgcolor-a">e</div>
</div>

<div class="choice-bar btn-toolbar" role="toolbar">
  <div class="btn-group" role="group">
    <button type="button" class="btn btn-default" id="reset-button">reset</button>
    <button type="button" class="btn btn-default" id="random-button">sample</button>
  </div>
  <div class="btn-group" role="group" id="button-choices"></div>
</div>

## Usage

- Press "sample" to get started
- The neural network is trained to predict the next character you'll write, if you write like Shakespeare
- The bar graph shows the confidence for the top 5 characters
- If you're not using a phone, you can type letters instead of pressing the buttons, and use "enter" instead of pressing the "sample" button
