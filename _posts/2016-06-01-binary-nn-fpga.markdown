---
layout: post
title: "Binary Neural Networks"
date: 2016-06-01 12:00:00
categories: machine-learning
excerpt: >
  This post provides a tutorial for implementing the BinaryConnect algorithm on an FPGA (currently a work-in-progress).
---

The Github repository for this project can be found [here](https://github.com/codekansas/binary-ml).

* TOC
{:toc}

# Introduction

BinaryConnect is an algorithm which was introduced in [Courbariaux et. al.](bengio) for training neural networks using binary weights, that is, weights that are one of two values. This is a smart thing to do for a couple of reasons:

 - Binary weights could act as a regularizer, like dropout, which could help reduce overfitting
 - Weights take much less memory to represent: Instead of using 32-bit floating point values, a single bit value can be used
 - Accumulate operations are faster than multiply operations (this will be discussed further later in this post)

For the coding examples, I will be using [Theano 0.8](http://deeplearning.net/software/theano/) (the bleeding edge version, as of this writing) and Python 2.7. If you have a problem running some code, send me an email about it at [bkbolte18@gmail.com](mailto:bkbolte18@gmail.com).

# Perceptron Learning Rule

The idea of binary operations has been interesting since the fledgling days of neural networks, starting with the perceptron learning rule. The output of a perceptron function depends on some set of weights, a bias and an input:

$$
\hat{y} = \left\{
        \begin{array}{ll}
            1 & \quad wx+b > 0 \\
            0 & \quad \text{otherwise}
        \end{array}
    \right.
$$

This can be reformulated as a matrix multiply and bias (MMB) operation followed by a nonlinear function, in this case the step function. The step function can be formulated as `y = lambda x: 0 if x < 0 else 1`. In Theano, this could be written `y = lambda x: T.switch(T.lt(x, 0), 0, 1)`. We can put together a perceptron which learns a decision boundary with the code below.

## Code Example

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

Fundamentally, the difference is that the nonlinear activation function does not allow us to use gradient descent on some cost function, because the derivative of the step function is zero. Even if we did have a cost function, the term `T.grad(cost, W)` would always be zero. This is the limitation of using binary values for a neural network; everything revolves around calculating gradients, and getting good gradients requires good floating point accuracy.

## Winner-Take-All

One strategy that has been used to approximate nonlinear functions in circuits is to use a winner-take-all (WTA) gate. [Wolfgang Maass](maass) demonstrated that WTA circuits are able to learn arbitrary continuous functions, using a learning rule much the same as the perceptron learning rule. However, this circuit is unable to learn highly complex manifolds such as those involved in necessary for object recognition, and more importantly cannot be stacked to increase complexity, due to the gradient descent issue.

<!-- Add links back to the top -->
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
[maass]: http://www.mitpressjournals.org/doi/abs/10.1162/089976600300014827?journalCode=neco#.V03BVZMrJE4