---
layout: post
title: Wireworld
date: 2017-10-01 12:00:00
categories: visualizations
excerpt: >
  A (slightly modified) Wireworld implementation
---

<script defer src="{{ site.cdn.d3js }}"></script>
<script defer src="/assets/posts/wireworld/main.js"></script>

<style>
.cell-selected {
  stroke: black;
  stroke-width: 1px;
}
.link-button {
  cursor: pointer;
  border: 1px solid black;
  padding: 1px 3px 1px 3px;
}
.connections {
  stroke: #52788b;
  stroke-width: 0.8;
}
svg#automaton {
  max-width: 80vw;
  max-height: 95vh;
  margin-bottom: 1em;
}
</style>

<svg viewBox="0 0 100 100" class="center-block no-select" id="automaton">
  <defs>
    <marker id="triangle" markerWidth="13" markerHeight="13" refx="1.5" refy="2" orient="auto">
      <path d="M2,0 a1,1 0 0,0 0,4" style="fill: #52788b;" />
    </marker>
  </defs>
</svg>

## Instructions

- This is a slightly modified implementation of a cellular automaton called [Wireworld](https://en.wikipedia.org/wiki/Wireworld)
- You can form connections by clicking on two cells
- You can click the same cell twice to change it's state
- You can <a class="link-button" id="clear-connections">click here</a> to clear the connections, or <a class="link-button" id="randomize-connections">click here</a> to randomize the connections, connecting nodes depth-first, or <a class="link-button" id="randomize-bfs">click here</a> to randomize the connections, connecting nodes breadth-first
- You can also <a class="link-button" id="toggle-sparking">toggle random sparking</a> on and off

## Rules

- There are three states: **conductor**, **head** and **tail**
- On each step, the states transition as follows:
  - **tail** turns into a **conductor**
  - **head** turns into a **tail**
  - **conductor** turns into a **head** if there are exactly one or two connecting **head** states
