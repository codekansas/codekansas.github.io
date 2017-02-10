---
layout: post
title: "Super-Resolving Images using GANs"
date: 2017-02-10 12:00:00
categories: machine-learning
excerpt: >
  Using Generative Adversarial Networks to super-resolve images.
---

{% include image.html url="/resources/gan/resolved_five.png" description="Increasing the resolution of MNIST digits. The first and second image are the training data; the first image is the downsampled MNIST digit, while the second image is the original MNIST digit. The subsequent images are super-resolved versions, with each step doubling the resolution." %}

# Introduction

In this post, I'll talk about a method for taking low-resolution images and increasing the resolution iteratively, using a Generative Adversarial Network. The backbone of this is a project I've been working on called [Gandlf][gandlf], which wraps around Keras models and makes it easy to build and train GANs without mucking about too much. There is a similar project which I found out about after I started working on Gandlf called [keras-adversarial][keras-adversarial].

# Related Links

 - [neural-enhance][neural-enhance]: A project which does basically the same thing as this post, written in Lasagne.
 - [Generating Large Images from Latent Vectors][hardmaru-mnist]: Train a model to take (X, Y) coordinates and output a pixel intensity, then interpolate between points to get high-resolution images.
 - [Pixel Recursive Super Resolution][dahl-pixelcnn]: Train a PixelCNN model to un-pixelate faces.

[neural-enhance]: https://github.com/alexjc/neural-enhance
[keras-adversarial]: https://github.com/bstriner/keras-adversarial
[gan-original]: https://arxiv.org/abs/1406.2661
[hardmaru-mnist]: http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/
[dahl-pixelcnn]: https://arxiv.org/pdf/1702.00783.pdf
[gandlf]: http://gandlf.org/
[gandlf-about]: http://gandlf.org/background/

