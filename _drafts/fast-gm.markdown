---
layout: post
title: "Accelerating Graphical Models using CUDA"
category: 🔬
excerpt: Describing a framework for optimizing graphical model algorithms in CUDA
math: true
---

In past posts, I described some optimizations for the [logsumexp function][logsumexp-post], as well as some tricks for writing [CUDA extensions][cuda-post] for PyTorch. In this post, I'll describe a framework I've been working on for accelerated inference on certain graphical models.

[logsumexp-post]: {% post_url 2020-05-20-logsumexp %}
[cuda-post]: {% post_url 2020-05-06-torch-cuda-tricks %}
