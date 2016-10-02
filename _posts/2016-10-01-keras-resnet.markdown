---
layout: post
title: "Analyzing Residual Networks using Keras"
date: 2016-09-30 12:00:00
categories: machine-learning
excerpt: >
  A tutorial on interpreting residual networks, as well as how to build them using Keras.
---

<table class="note">
<tr><th>Github Repository</th></tr>
<tr><td>
The repository associated with this post can be found <a
href="https://github.com/codekansas/keras-resnet"
target="_blank">here</a>.
</td></tr>
</table>

* TOC
{:toc}

# Introduction

Residual networks are one of the hot new ways of thinking about neural networks, ever since they were used to win the ImageNet competition in 2015. ResNets were originally introduced in the paper [Deep Residual Learning for Image Recognition][sun-resnet-paper] by He et. al. The remarkable thing about the ResNet architecture is just how crazy deep it is. For comparison, the Oxford Visual Geometry Group released a [Very Deep Convolutional Network for Large-Scale Visual Recognition][vgg16], which even has "Very Deep" in the name, and it had either 16 or 19 layers. ResNet architectures were demonstrated with 50, 101, and even *152* layers. More surprising than dectupling the number of layers of another architecture, the deeper ResNet got, the more its performance grew. It [did very well][imagenet2015] in the 2015 ImageNet competition, and seems to be the best single model out there for object recognition, with most of the [2016 ImageNet models][imagenet2016] being ensembles of other models.

# What are residuals?

First, here is a graphic illustrating the concept of a "residual network".

![ResNet general architecture](/resources/resnet/resnet_general.png)

Note that this "network" is really just a building block. In the ResNet architecture, a whole bunch of these were stuck together.

Mathematically, we can express this network with input $$x$$, some transformation $$f(x)$$ (the complicated network in the diagram), some merge function $$m(x, y)$$, some activation function $$a(x)$$ and output $$y$$ as

$$y = a(m(x, f(x)))$$

Characteristically, $$f(x)$$ is a convolution or series of convolutions, $$m(x, y)$$ is simply addition, and $$a(x)$$ is the rectified linear unit activation function, giving us the equation

$$y = \text{relu}(x + f(x))$$

The **residual** is the network error that we want to correct at a particular layer.

When people talk about "ResNet" as an abbreviaion for "Residual Network", it is usually to refer to *convolutional* neural networks. But in principle, the idea behind them can be applied to any type of neural network. For example, two researchers at Google recently used residuals applied to a Gated Recurrent Unit for [image compression][image-compression].

# Why are residuals a good idea?

The central idea of the ResNet paper is that it is a good idea, when adding more layers to a network, to keep the representation more or less the same. In other words, extra layers shouldn't warp the representation very much. Suppose a shallow network perfectly represents the data, and more layers are added. Since the shallow network works perfectly, the best thing for the new layers to do would be to learn the identity function. If the shallow network made a few errors, we would want the new layers to learn to correct the errors, but otherwise not affect the output very much.

Phrased another way, it is an easier learning problem if the network learns to correct the *residual* error. Once a good representation is learned, the network shouldn't mess with it too much. The other side to this problem is that we want the shallow network to be able to learn a good solution, without having to learn gradients through higher level layers.

## Comparison with Highway Networks

This idea has been phrased differently as **information flow**, and shows up in [LSTM and GRU networks][colah-lstm] and [Highway networks][highway-schmid]. The key difference between residual and highway networks is the absence of gating. In a highway network, the merge function $$m(x, y)$$, which for the residual network was simply addition, would instead be expressed as

$$m(x, f(x)) = x * g(x) + f(x) * (1 - g(x))$$

where $$g(x)$$ is a gating function dependent on $$x$$. Depending on how the problem is formulated, the gating function can significantly increate the number of parameters. Aside from that, this formulation might get in the way of the idea discussed earlier; we want the lower layers to learn a near-perfect representation, so we should avoid modifying this representation at all in upper layers.

![ResNet information flow](/resources/resnet/resnet_infoflow.png)

The graphic above illustrates how the residual network achieves better information flow by passing more information through identity mapping to avoid going through the residual function.

## Mathematical Example

As a case study, let's consider a two-layer fully-connected network. Given input vector $$\vec{x}$$ and output vector $$\vec{y}$$, a feedforward network can be expressed as

$$h_n(\vec{u}) = \text{sigm}(\text{dot}(\vec{u}, \mathbf{W}_n) + \vec{b}_n)$$

$$\vec{y}_1 = h_2(h_1(\vec{x}))$$

We can instead express the second layer as a residual:

$$g_n(\vec{u}) = \text{sigm}(\text{dot}(\vec{u}, \mathbf{W}_n) + \vec{b}_n + \vec{u})$$

$$\vec{y}_2 = g_2(h_1(\vec{x}))$$

Here's a visual representation of these equations:

![Fully connected network and residual](/resources/resnet/feedforward_resnet.png)

Allegedly, the key advantage of doing this is so that more information can flow from the output of the first layer to the end result. Let's take advantage of some loose mathematical notation to characterize what we mean by "information flow". Intuitively, high information flow between two parameters means that changing one parameter significantly affects the other parameter. In other words, the partial derivative (usually called a "gradient") is relatively large.

For both networks:

$$\frac{\partial \vec{y}}{\partial h_1} = \frac{\partial h_2}{\partial h_1} = \frac{\partial \text{sigm}}{\partial h_1}$$

For the first network, the term $$\frac{\partial \text{sigm}}{\partial h_1}$$ has to go through the weight matrix $$W_2$$. When the output of the first layer is dotted with the weight matrix to get the output, it means that the gradient suddenly depends on all of the parameters in the weight matrix. Mathematically:

$$\frac{\partial \vec{y}_1}{\partial h_1} = \frac{\partial \text{sigm}}{\partial \text{dot}} \frac{\partial \text{dot}}{\partial h_1}$$

However, for the second network, there are two routes it can go, with one of them avoiding the weight matrix completely:

$$\frac{\partial \vec{y}_1}{\partial h_1} = \frac{\partial \text{sigm}}{\partial \text{dot}} \frac{\partial \text{dot}}{\partial h_1} + \frac{\partial \text{sigm}}{\partial h_1}$$

To take the concept of information flow a step further, we can use a rectified linear unit instead of a sigmoid for our activation function. If we use ReLUs and stack multiple layers together, any positive outputs of any layer are passed along completely, which is really good information flow.

# Using Keras to Build a Residual Network

This concept doesn't seem too hard to actually implement. I'll go ahead and assume if you decided to read this blog post, you're comfortably familiar with Keras and don't need someone to hype it up to you.

As of this writing, Keras supports two wrappers, `Bidirectional` and `TimeDistributed`. The first one lets you take a recurrent neural network model and make it bidirectional, meaning it will traverse your input data in the forward and backward directions. The second one lets you take a model which has no time component (for example, a `Dense` feedforward network) and apply it over a series of timesteps.

This is another wrapper, which takes a model that has the same input and output dimensions and applies the residual. I'll dump the code here, and explain it below. The code is also available [here](https://github.com/codekansas/keras-resnet/blob/master/resnet.py).

{% highlight python %}
from __future__ import absolute_import

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Wrapper, Merge


class Residual(Wrapper):
    """This wrapper automatically applies a residual to a model.

    For an input `x` and a model `F(x)`, the residual wrapper gives the output
    `y = x + F(x)`. In this configuration, the output of F(x) must have the
    same shape as x. Other merge modes are supported besides summation.

    ```python
        input = Input(shape=(5,))

        # Apply the residual normally
        output1 = Residual(Dense(5), merge_mode='sum')(input)

        # Throws an exception due to mismatching shapes
        output2 = Residual(Dense(3), merge_mode='sum')(input)

        # Product: `y = x * F(x)`
        output3 = Residual(Dense(5), merge_mode='mul')(input)
    ```

    For more modes, see: https://keras.io/layers/core/#merge

    Alternatively, a function which takes the input and the layer output
    can be passed to define the merge:

    ```python
        from keras.layers import Merge
        def diff_merge():  # x_fx = [x, fx]
            diff = lambda x: x[1] - x[0]
            return Merge(mode=diff, output_shape=lambda x: x)

        # Difference: `y = F(x) - x`
        output4 = Residual(Dense(5), merge_mode=diff_merge())(input)
    ```

    Arguments:
        layer: The layer to wrap
        merge_mode: The merge operation
    """
    def __init__(self, layer, merge_mode='sum', **kwargs):
        self.merge_mode = merge_mode
        self.supports_masking = True
        super(Residual, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        output_shape = self.layer.get_output_shape_for(input_shape)
        if output_shape != input_shape:
            raise Exception('Cannot apply residual to layer "{}": '
                            'mismatching input and output shapes'
                            '"{}" and "{}"'
                            .format(self.layer.name, input_shape, output_shape))
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.input_spec = [InputSpec(shape=input_shape)]
        output_shape = self.layer.get_output_shape_for(input_shape)
        if self.merge_mode == 'weighted':
            self.U = K.random_uniform_variable(output_shape[1:], 0, 1,
                                               name='{}_U'.format(self.name))
        super(Residual, self).build()

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        layer_output = self.layer.call(x, mask)
        if isinstance(self.merge_mode, str):
            self.merge_mode = Merge(mode=self.merge_mode)
        output = self.merge_mode([x, layer_output])
        return output
    
    @classmethod
    def from_config(cls, config):
        from keras.utils.layer_utils import layer_from_config
        merge_mode = layer_from_config(config.pop('merge_mode'))
        residual = super(Residual, cls).from_config(config)
        residual.merge_mode = merge_mode
        return residual
    
    def get_config(self):
        config = {"merge_mode": {'class_name': 'Merge',
                                 'config': self.merge_mode.get_config()}}
        base_config = super(Residual, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
{% endhighlight %}

[highway-schmid]: https://arxiv.org/abs/1505.00387
[colah-lstm]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
[image-compression]: https://research.googleblog.com/2016/09/image-compression-with-neural-networks.html
[imagenet2016]: http://image-net.org/challenges/LSVRC/2016/results
[imagenet2015]: http://image-net.org/challenges/LSVRC/2015/results
[vgg16]: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
[sun-resnet-paper]: https://arxiv.org/abs/1512.03385
