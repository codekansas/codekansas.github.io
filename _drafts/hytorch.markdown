---
layout: post
title: "HyTorch: A Bare-Bones PyTorch Compiler"
category: 🔬
excerpt: An exploration of functional PyTorch and neural network compilation.
---

I noticed after writing this that it was getting pretty long, so I've added links to the various sections below.

- [Motivation](#motivation)
- [How do you compile a neural network?](#how-do-you-compile-a-neural-network)
  - [Tracing Compiler](#tracing-compiler)
  - [Scripting Compiler](#scripting-compiler)
  - [Motivating Example](#motivating-example)
- [A simple neural network in Lisp](#a-simple-neural-network-in-lisp)

## Motivation

As more people decide that they need to make [their own][make-your-own-accelerator-chip] accelerators for handling their deep learning workflows, a problem has started to surface in the PyTorch research-to-production pipeline. The preferred method for deploying a trained neural network, at least at Facebook, is to convert it to [TorchScript][torchscript], load the generated binary with [LibTorch][libtorch], and write your C++ server to serve requests. The reason that this is the best-supported deployment pipeline is because that is the pipeline that is most common at Facebook, and naturally they have a need to provide solid support.

However, TorchScript can be quite restrictive, if not extremely unwieldy, for handling other kinds of workflows. **In this post**, I will:

1. Write a simple PyTorch compiler
2. Train a binarized neural network on MNIST
3. Do some light-weight [neural architecture search][neural-architecture-search-wiki]
4. Compile the model with the custom compiler
5. Run it in a completely LibTorch-free environment

While this certainly won't be _production-quality_, I think it will illustrate some interesting ideas about functional programming and neural network compilation.

## How do you compile a neural network?

When I sat down to write this, my understanding of compilers was pretty much limited to a course I took in college on the subject. For the first class project, we wrote a LISP compiler in LISP. It was a pretty fun challenge, based on the fact that LISP can easily inspect itself. As my professor described it, LISP is essentially as if someone took an [Abstract Syntax Tree][ast-wiki] and made it into a programming language. There is a reason it's the [second-oldest][lisp-wiki] programming language - the rest of compiler theory is basically just applied LISP!

![LISP][xkcd-lisp-comic]
_As a student who learned computer science in the post-Python era I didn't understand this comic until I took a compilers class, but I still found it funny. ([Reference][xkcd-lisp-comic-reference])_

I decided to use this as a jumping off point. When I've worked on neural network compilation in other projects, it typically starts at some level which is defined by existing, moderately-esoteric tools, and I wanted to see if I could start totally from scratch. And anyway, coming up with a good representation for neural networks is far from being a solved problem. Here are a few example representations that different organizations have implemented:

1. [ONNX][onnx-main]: Open Neural Network Exchange, an open standard for machine learning interoperability, which supports a large number of neural network training libraries, including Keras, PyTorch, and Chainer.
2. [TensorFlow XLA][tensorflow-xla]: Accelerated Linear Algebra, a domain-specific compiler for linear algebra. This is specific to TensorFlow, but it shows quite promising performance boosts and is used for training on Google's TPUs. They have well-documented [operation semantics][tensorflow-xla-semantics].
3. [Apache TVM][apache-tvm]: Tensor Virtual Machine (although annoyingly it is impossible to find out what TVM stands for from their Github page or main landing page). An open-source compiler stack that supports a number of backends. I've never used TVM personally, but from a cursory glance at their docs it seems like it is more of a backend to ONNX than an intermediate representation of its own.
4. [Glow][glow-compiler]: A Facebook-supported compiler stack for different hardware backends (although I haven't heard much about it lately).
5. PyTorch has a few different intermediate representations that are packaed with it out-of-the-box, including:
   - [TorchScript][torchscript], which converts a subset of the [Python AST][python-ast] to a new intermediate representation. This IR is not very easily accessible, but it is well-supported with [LibTorch][libtorch].
   - [Torch FX][torch-fx], a currently-in-beta symbolic tracer, intermediate representation, and Python code generation tool that is intended to be much more user-friendly than TorchScript for downstream users wanting to do programmatic manipulations on their PyTorch graphs (including good tools for code generation).
6. [TensorRT][tensor-rt-api]: A framework for doing accelerated inference from Nvidia

There are a lot more, but these are some of the main ones that come to mind. One way of categorizing these compilers is to split them into two camps, which I'm going to call the **tracing** camp and the **scripting** camp, because those are the two modes that TorchScript supports.

### Tracing Compiler

The **tracing** camp essentially takes some dummy tensors and runs them through the model you want to compile, keeping track of the operations performed on those tensors. Besides TorchScript's tracing mode, this category includes [ONNX][onnx-main] and [Torch FX][torch-fx]. This mode has several advantages:

- It is easy to grok
- It is relatively bug-proof, because you can check that the tensors you generate from your IR match the original tensors from the model
- It handles most types of neural networks, like ResNet, and many neural networks can be made to work with it

However, there are some things that tracing can't do:

- It can't handle control flow variables, like loops or if statements, unless you can come up with a way to hack around this
  - Because of this, the generated model graph will likely be larger, which may not be ideal if you are in a program-memory-constrained environment
- It is generally slower, because you have to actually run the model, rather than just looking at the code

### Scripting Compiler

The **scripting** camp, instead of passing tensors through the model, comes up with an intermediate representation of the network just based on the operations it contains. Besides TorchScript's scripting mode, this category includes [TensorFlow XLA][tensorflow-xla]. This mode has several advantages:

- It works with control flow, which can reduce the generated model graph size
- It is more robust to different inputs, because you are tracing the ops, rather than relying on one dummy input to tell you everything you need to know about the model

However, this mode is more difficult to implement, and probably more bug-prone, simply because it has to support a wider variety of use cases.

### Motivating Example

To demonstrate the difference in functionality between the above two compilation modes, suppose we want to train a simple recurrent neural network. As input, we'll take a length `N` list of tensors, and as output we want the last tensor.

````python
class TestModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.hid_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.in_to_hid = nn.Linear(input_dim, hidden_dim)
        self.hid_to_hid = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        hid: Tensor = self.hid_init.repeat(inputs[0].shape[0], 1)
        for x in inputs:
            hid = torch.sigmoid(self.hid_to_hid(hid) + self.in_to_hid(x))
        return hid
```

We can test that this model is working as expected by writing some boilerplate code:

```python
batch_dim, input_dim, hidden_dim, num_steps = 4, 16, 32, 10
input_shape = (batch_dim, input_dim)
model = TestModel(input_dim, hidden_dim)
test_inputs = [torch.randn(*input_shape) for _ in range(num_steps)]
test_output = model(test_inputs)
assert test_output.shape == (batch_dim, hidden_dim)
```

Let's run this model through the tracing and scripting compilation process.

```python
traced_model = torch.jit.trace(model, (test_inputs,))
test_traced_output = traced_model(test_inputs)
assert test_traced_output.shape == (batch_dim, hidden_dim)

scripted_model = torch.jit.script(model)
test_scripted_output = scripted_model(test_inputs)
assert test_scripted_output.shape == (batch_dim, hidden_dim)

assert (test_traced_output == test_scripted_output).all()
```

As we can see from the final check, the two compiled models are performing the exact same computations, at least when we pass them both the expected inputs. However, there is a huge difference in what they look like under the hood. Here is what the **scripted** model looks like:

```python
# print(scripted_model.code)

def forward(self,
    inputs: List[Tensor]) -> Tensor:
  hid = torch.repeat(self.hid_init, [(torch.size(inputs[0]))[0], 1])
  hid0 = hid
  for _0 in range(torch.len(inputs)):
    x = inputs[_0]
    _1 = torch.add((self.hid_to_hid).forward(hid0, ), (self.in_to_hid).forward(x, ))
    hid0 = torch.sigmoid(_1)
  return hid0
```

And here is what the **traced** model looks like:

```python
# print(traced_model.code)

def forward(self,
    inputs: List[Tensor]) -> Tensor:
  _0 = self.in_to_hid
  _1 = self.hid_to_hid
  _2 = self.hid_init
  input, input0, input1, input2, input3, input4, input5, input6, input7, input8, = inputs
  _3 = ops.prim.NumToTensor(torch.size(input, 0))
  input9 = torch.repeat(_2, [int(_3), 1])
  _4 = torch.add((_1).forward(input9, ), (_0).forward(input, ))
  input10 = torch.sigmoid(_4)
  _5 = torch.add((_1).forward1(input10, ), (_0).forward1(input0, ))
  input11 = torch.sigmoid(_5)
  _6 = torch.add((_1).forward2(input11, ), (_0).forward2(input1, ))
  input12 = torch.sigmoid(_6)
  _7 = torch.add((_1).forward3(input12, ), (_0).forward3(input2, ))
  input13 = torch.sigmoid(_7)
  _8 = torch.add((_1).forward4(input13, ), (_0).forward4(input3, ))
  input14 = torch.sigmoid(_8)
  _9 = torch.add((_1).forward5(input14, ), (_0).forward5(input4, ))
  input15 = torch.sigmoid(_9)
  _10 = torch.add((_1).forward6(input15, ), (_0).forward6(input5, ))
  input16 = torch.sigmoid(_10)
  _11 = torch.add((_1).forward7(input16, ), (_0).forward7(input6, ))
  input17 = torch.sigmoid(_11)
  _12 = torch.add((_1).forward8(input17, ), (_0).forward8(input7, ))
  input18 = torch.sigmoid(_12)
  _13 = torch.add((_1).forward9(input18, ), (_0).forward9(input8, ))
  return torch.sigmoid(_13)
```

We can illustrate illustrate the weaknesses of the traced model by passing it a different size list as input. The scripted model handles the new inputs without any issues:

```python
new_test_inputs = [torch.randn(*input_shape) for _ in range(num_steps - 1)]
new_test_scripted_output = scripted_model(new_test_inputs)
assert new_test_scripted_output.shape == test_scripted_output.shape
```

However, when we pass the same inputs to the traced model:

```python
new_test_traced_output = traced_model(new_test_inputs)
```

We get `RuntimeError: Expected 10 elements in a list but found 9`

## A simple neural network in Lisp

For the purposes of this blob post, I'm going to tackle the simplest neural network task around, [MNIST][mnist-yann-lecun-website]. Specifically, I'll be referencing the [example model][mnist-pytorch] from the PyTorch repo. For reference, here is what that model looks like (per their implementation):

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

To implement this model in a functional programming language, I'm going to use the [Hy][hylang-docs] language, which, per their website, is "a Lisp dialect that's embedded in Python". I'm not going to go in to the details of the language syntax (although I highly recommend browsing the [Hy API][hylang-api]). Here is my parallel implementation of the above model in Hy (using the `->` [threading macro][hy-threading-macro]):

```hy
(import [torch [nn Tensor]] [torch.nn [functional :as F]])

; Defines a simple two-layer neural network.
(defclass Net [nn.Module]
    (defn __init__ [self]
        ((. (super) __init__))
        (setv self.conv1 (nn.Conv2d 1 32 3 1))
        (setv self.conv2 (nn.Conv2d 32 64 3 1))
        (setv self.dropout1 (nn.Dropout 0.25))
        (setv self.dropout2 (nn.Dropout 0.5))
        (setv self.fc1 (nn.Linear 9216 128))
        (setv self.fc2 (nn.Linear 128 10)))
    (defn ^Tensor forward [self ^Tensor x]
        (-> x
            self.conv1
            F.relu
            self.conv2
            F.relu
            (F.max_pool2d 2)
            self.dropout1
            (torch.flatten 1)
            (self.fc1)
            (F.relu)
            (self.dropout2)
            (self.fc2)
            (F.log_softmax :dim 1))))
```

We can verify that it is, in fact, a parallel implementation by running the useful `hy2py` command to generate Python code:

```python
import hy
from torch import nn, Tensor
from torch.nn import functional as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) ->Tensor:
        return F.log_softmax(self.fc2(self.dropout2(F.relu(self.fc1(torch.
            flatten(self.dropout1(F.max_pool2d(F.relu(self.conv2(F.relu(
            self.conv1(x)))), 2)), 1))))), dim=1)
```

There's some notable differences between the generated Python code and the original implementation (namely that I've added types to the `forward` function, and the use of the placeholder `x` variable is gone), but it is functionally identical, which is the best kind of identical.

[neural-architecture-search-wiki]: https://en.wikipedia.org/wiki/Neural_architecture_search
[hylang-api]: https://docs.hylang.org/en/stable/language/api.html
[hylang-docs]: https://docs.hylang.org/en/alpha/
[hy-threading-macro]: https://docs.hylang.org/en/stable/language/api.html#id3
[mnist-pytorch]: https://github.com/pytorch/examples/blob/master/mnist/main.py
[mnist-yann-lecun-website]: http://yann.lecun.com/exdb/mnist/
[xkcd-lisp-comic]: https://imgs.xkcd.com/comics/lisp.jpg
[xkcd-lisp-comic-reference]: https://xkcd.com/224/
[lisp-wiki]: https://en.wikipedia.org/wiki/Lisp_(programming_language)
[ast-wiki]: https://en.wikipedia.org/wiki/Abstract_syntax_tree
[torch-fx]: https://pytorch.org/docs/stable/fx.html
[python-ast]: https://docs.python.org/3/library/ast.html
[glow-compiler]: https://ai.facebook.com/tools/glow/
[apache-tvm]: https://tvm.apache.org/
[tensorflow-xla]: https://www.tensorflow.org/xla
[tensorflow-xla-semantics]: https://www.tensorflow.org/xla/operation_semantics
[onnx-main]: https://onnx.ai/
[intermediate-representation-wiki]: https://en.wikipedia.org/wiki/Intermediate_representation
[libtorch]: https://pytorch.org/cppdocs/installing.html
[torchscript]: https://pytorch.org/docs/stable/jit.html
[make-your-own-accelerator-chip]: https://towardsdatascience.com/how-to-make-your-own-deep-learning-accelerator-chip-1ff69b78ece4
[tensor-rt-api]: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/
````
