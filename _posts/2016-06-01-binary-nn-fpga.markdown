---
layout: post
title: "BinaryConnect on an FPGA"
date: 2016-06-01 12:00:00
categories: ml
---

This post provides a tutorial on implementing the BinaryConnect algorithm on an FPGA. It is currently a work in progress, so I will be adding sections as I work on them (I find that explaining things in words helps me clarify my thought process some). Hopefully I'll finish this in about a month. The Github repository where I will be adding code is located [here](https://github.com/codekansas/binary-ml).

* TOC
{:toc}

# Introduction

BinaryConnect is an algorithm which was introduced in [Courbariaux et. al.](bengio) for training neural networks using binary weights, in other words, weights that are one of two values. This is a smart thing to do for a couple of reasons:

 - Binary weights could act as a regularizer, like dropout, which could help reduce overfitting
 - Weights take much less memory to represent: Instead of using 32-bit floating point values, a single bit value can be used
 - Accumulate operations are faster than multiply operations (this will be discussed further later in this post)

# Some Foundations of Neural Networks

This section will probably seem very elementary, but could offer a new way of looking at neural networks. The idea of binary operations has been interesting since the fledgling days of neural networks, starting with the perceptron learning rule. The output of a perceptron function depends on some set of weights, a bias and an input:

$$
\hat{y} = \left\{
        \begin{array}{ll}
            1 & \quad wx+b > 0 \\
            0 & \quad \text{otherwise}
        \end{array}
    \right.
$$

This can be reformulated as a matrix multiply and bias (MMB) operation followed by a nonlinear function, in this case the step function. The step function can be formulated as `y = lambda x: 0 if x < 0 else 1`. In Theano, this could be written `y = lambda x: T.switch(T.lt(x, 0), 0, 1)`. We can put together a perceptron which learns a decision boundary with the code below.

{% highlight python %}
import theano
import theano.tensor as T

nonlinearity = lambda x: T.switch(T.lt(x, 0), 0, 1)

X = T.matrix(name='X', dtype=theano.config.floatX)
y = T.matrix(name='y', dtype=theano.config.floatX)
lr = T.scalar(name='learning rate', dtype=theano.config.floatX)

def get_weights(name, *shape):
    return theano.shared(np.random.randn(*shape), strict=False, name=name)

W = get_weights('W', n_in, n_out)

y_hat = nonlinearity(T.dot(X, W) - 1)

updates = [(W, W - lr * T.dot((y_hat - y).T, X).T)]
train = theano.function([X, y, lr], [], updates=updates)
{% endhighlight %}

This code can be used to learn an arbitrary linear decision boundary, as seen in the following figure (the red line indicates the initial decision boundary, while the blue line indicates the decision boundary after training for 1000 epochs).

![Perceptron updates](/resources/binary_ml/perceptron_updates.png)

Note the way that weights are updated in this code.

{% highlight python %}
updates = [(W, W - lr * T.dot((y_hat - y).T, X).T)]
{% endhighlight %}

This is the perceptron learning rule. Conceptually, it looks at any training points which are on the wrong side of the decision boundary and moves the decision boundary a little bit in a direction which would put them on the correct side. Note how this contrasts with a typical implementation of an update rule when building a neural network, which might look like:

{% highlight python %}
updates = [(W, W - lr * T.grad(cost, W))]
{% endhighlight %}

Fundamentally, the difference is that the nonlinear activation function does not allow us to use backpropagation, because the derivative of the step function is zero. This is the limitation of using binary values for a neural network; everything revolves around calculating gradients.

<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.2.2/jquery.min.js"></script>
<script type="text/javascript">
// Turn all headers into links back to the table of contents
$(document).ready(function() {
    $("article").find("h1, h2, h3, h4, h5, h6").each(function(index) {
        var content = $(this).text();
        $(this).html("<a href=\"#markdown-toc\" style=\"color: black;\">" + content + "</a>");
    });
});
</script>

[bengio]: http://arxiv.org/pdf/1511.00363v3.pdf
[original code]: https://github.com/MatthieuCourbariaux/BinaryConnect