---
layout: post
title: "A Neural Network in 28 Lines of Theano"
date: 2016-02-23 12:00:00
categories: machine-learning
excerpt: >
  A quick introduction to using Theano for deep learning.
image: /resources/index/theano.jpeg
links:
 - View Gist: https://gist.github.com/codekansas/87dd63ca4e2286e332c7967520ce143c#file-theano_two_layer-py
---

This tutorial is a bare-bones introduction to Theano, in the style of [Andrew Trask's Numpy example][trask]. For a more complete version, see the [official tutorial][theano-tut]. It is mostly to help me learn to use Theano, and feedback is more than welcome.

I used Python 3.5 and Theano 0.8. If you already have Theano set up, skip this. Otherwise, see the installation instructions [here][theano-install]; usually this means doing a `pip install Theano`.
<h2>Straight Code</h2>
Here is just the code. The network has 5 hidden neurons and learns the XOR function, which takes two inputs and returns a high output only if exactly one of the inputs is high. Otherwise, it returns a low output.

<script src="https://gist.github.com/codekansas/87dd63ca4e2286e332c7967520ce143c.js"></script>

<h2>Explanation</h2>
Ok, let's see what's going on here.

{% highlight python %}
X = theano.shared(value=np.asarray([[1, 0], [0, 0], [0, 1], [1, 1]]), name='X')
y = theano.shared(value=np.asarray([[1], [0], [1], [0]]), name='y')
rng = np.random.RandomState(1234)
LEARNING_RATE = 0.01
{% endhighlight %}

Here, we're creating shared variables X and y, representing our inputs and outputs, respectively. [Shared variables][theano-shared] are like global variables in a programming language; they are shared between functions, such as the functions "train" and "test" later on. We also initialize a random number generator "rng" and define a learning rate.

{% highlight python %}
def layer(n_in, n_out):
    return theano.shared(value=np.asarray(rng.uniform(low=-1.0, high=1.0,
    	   size=(n_in, n_out)), dtype=theano.config.floatX), name='W', borrow=True)

W1 = layer(2, 3)
W2 = layer(3, 1)
{% endhighlight %}

Here, we define a function which creates and returns a matrix of random numbers between -1.0 and 1.0, whose size we specify. The matrix is also a shared variable. We use this function to create the weights W1 and W2 for our network.

{% highlight python %}
output = T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(X, W1)), W2))
cost = T.sum((y - output) ** 2)
updates = [(W1, W1 - LEARNING_RATE * T.grad(cost, W1)),
           (W2, W2 - LEARNING_RATE * T.grad(cost, W2))]
{% endhighlight %}

We finally get into constructing the network. Theano usefully includes the "sigmoid" function, which is used as the network's activation function. We multiply the input vector X by the first weight matrix and apply the activation function; we then take this output and multiply it by the second weight matrix before again applying the activation function.

For the neural network, we would like to minimize the squared error of the network, which is shown in our "cost" function. The squared error is the difference between the output of the network and the desired output. Since this is a binary classification task, we have one output, 0 or 1. If the XOR function accepts, we would like the network to output a 1; otherwise, output a 0.

The last part, "updates", defines how we want to change our network on each update step. We do this by trying to minimize the cost function with respect to the weights. This can be done with [stochastic gradient descent][sgd-wiki]; we calculate the gradient of the cost function with respect to the weights, and change the weights in the direction that causes the cost function to go down. Theano does this for us, using the "grad" function.

As an aside, because of the way we defined our weight matrices, the first multiplication / activation function increases the dimensionality of the input vector to 3 dimensions. This is important for allowing the network to learn the XOR function. [This post][manifold-blog] provides some intuition about why this is the case. For many applications, however, we are more concerned with reducing the dimensionality of our input vector.

{% highlight python %}
train = theano.function(inputs=[], outputs=[], updates=updates)
test = theano.function(inputs=[], outputs=[output])

for i in range(60000):
    if (i+1) % 10000 == 0:
        print(i+1)
    train()

print(test())
{% endhighlight %}

Here, we define our "train" and "test" functions. The "train" function updates the weights according to the update rules we provided earlier, after calculating the cost function. The "test" function gives us the output of the network. We then run the network through 60000 training steps. After training, we print the output, and lo and behold, it approximates our XOR function pretty well!

[trask]: https://iamtrask.github.io/2015/07/12/basic-python-network
[theano-tut]: http://deeplearning.net/tutorial/gettingstarted.html]
[theano-install]: http://deeplearning.net/software/theano/install.html
[theano-shared]: http://deeplearning.net/software/theano/library/compile/shared.html
[code-src]: https://github.com/codekansas/ml/blob/master/theano_stuff/two_layer.py
[manifold-blog]: https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
[sgd-wiki]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
