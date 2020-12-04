---
layout: post
title: "Optimized Log-Sum-Exp PyTorch Function"
category: 🔬
excerpt: A walkthrough of how to optimize the log-sum-exp function in PyTorch.
math: true
---

I've recently been working on writing a CUDA kernel for a project I've been working on. I haven't done much CUDA programming before, but it's been an interesting journey that I thought would be a useful exercise to share with other people. I'm going to assume some familiarity with PyTorch and the general idea of what CUDA is.

If you're like me, you probably use PyTorch or Tensorflow as a deep learning API, and never really think too much about what is happening under-the-hood. I think this is a testament to how great these APIs are, and much they improve productivity. However, that makes it easy to forget that these APIs are under active development, and in fact there is some low-hanging fruit, performance-wise. PyTorch's `logsumexp` is a good example of a function which is used liberally for some applications which it is not optimal for.

This idea was largely inspired by [this repo][genbmm] from Harvard NLP, which provided a kernel for speeding up the log-sum-exp part of a CRF or HMM model. I was inspired to investigate this in greater detail.

## Introduction

In [past posts][prev-post] I've given an introduction to the **forward-backward algorithm** and the **Viterbi algorithm**, two algorithms which are used for performing inference in Hidden Markov Models and Conditional Random Fields. In this post, I'm going to talk about one of the core concepts for making these models work, which is **log-space operations**. Doing inference in these models usually involves multiplying together very small numbers a large number of times, which can quickly become computationally intractable. Double-precision numbers are [stored][double-precision] in 64 bits using 1 bit to represent the sign, 11 bits for the exponent, and 52 bits for the fraction.

{% include /images/logsumexp/precision.svg %}

This means we can exceed the precision of the exponent relatively easily, if our exponent cannot be represented in `2^11 = 2048` bits. Consider the simple C++ program below, where we naively compute the value

$$\frac{2^{2^{12}}}{2^{1 + 2^{12}}} = \frac{1}{2}$$

by computing the numerator and denominator, then dividing.

{% highlight cpp %}
#include <stdio.h>

using namespace std;

int main() {
    double v = 2.0;
    for (int i = 1; i <= 12; i++)
        v *= v;  // v = 2 ^ (2 ^ i)
    double num = v, denom = v * 2;
    printf("num: %e, denom: %e, result: %e\n", num, denom, num / denom);
    return 0;
}
{% endhighlight %}

We know the result should be `0.5`. However, when we run it, it prints out the following:

{% highlight bash %}
num: inf, denom: inf, result: -nan
{% endhighlight %}

### Log-Space Operations

Because performing these repeated multiplications can lead to this underflow problem relatively easily for models such as CRFs and HMMs, it behoves us to find a more numerically stable solution. In this case, we can take advantage of the following identities:

$$
\begin{aligned}
x' & = \log(x) \\
x & = \exp(x') \\
x_1x_2 ... x_n & = \exp(x'_1 + x'_2 + ... + x'_n)
\end{aligned}
$$

Instead of having to perform the numerically unstable multiplications, we can perform numerically stable additions on the logs of these values, and apply the exponential function once we're done. If we want to perform additions on the logs of these values in a numerically stable way, we can naively do the following:

$$x_1 + x_2 + ... + x_n = \exp(x'_1) + \exp(x'_2) + ... + \exp(x'_n)$$

However, if the left side of this equation is once again very large (assuming we are going to divide it by something else later), this can lead to unwanted overflow. Instead, in practice, it is better to use the identity below, which is known as the `log-sum-exp` function. By subtracting the max value out out of each of the components of the addition, we can usually keep the exponential part from blowing up too much.

$$
\begin{aligned}
x^* & = \max(x'_1, x'_2, ..., x'_n) \\
x_1 + x_2 + ... x_n & = \exp(x^* + \log(\exp(x'_1 - x^*) + ... + \exp(x'_n - x^*)))
\end{aligned}
$$

We can now re-write our C++ program from earlier:

{% highlight cpp %}
#include <stdio.h>
#include <math.h>

using namespace std;

int main() {
    double v = log(2.0);
    for (int i = 1; i <= 12; i++)
        v += v;  // v = 2 ^ (2 ^ i)
    double num = v, denom = v + log(2.0);
    printf("log(num): %e, log(denom): %e, result: %e\n", num, denom, exp(num - denom));
    return 0;
}
{% endhighlight %}

This results in the correct answer.

{% highlight bash %}
log(num): 2.839131e+03, log(denom): 2.839824e+03, result: 5.000000e-01
{% endhighlight %}

In some literature, this is known as the [log semiring][log-semiring]; in particular, ([Goodman 1999][semiring-paper]) showed how a number of common algorithms can simply be thought of as derivatives of value functions computed over different semirings (it's a really interesting mathematical paper and a great way to conceptualize CRFs and HMMs).

## Mathematical Formalism

To provide some mathematical formalism for the examples above, it's important to expand on the semiring concept. It's actually pretty straight-forward, even if it sounds a bit complicated at first. The pair of functions (`sum`, `logsumexp`) is an example of a [semiring][log-semiring], meaning that it generalizes the multiplication and addition functions. This is some mathematical jargon which is easier to explain with an example. Lets define two semirings:

$$
\begin{aligned}
a \oplus_{\text{normal}} b & = a + b\\
a \otimes_{\text{normal}} b & = a b \\
a \oplus_{\text{log}} b & = \text{logsumexp}(a, b) \\
a \otimes_{\text{log}} b & = a + b
\end{aligned}
$$

We can then switch our operations between the two semirings:

$$
\begin{aligned}
a \oplus_{\text{normal}} b & = \exp(\log a \oplus_{\text{log}} \log b) \\
a \otimes_{\text{normal}} b & = \exp(\log a \otimes_{\text{log}} \log b)
\end{aligned}
$$

This is the heart of what we're doing. Since the log semiring is much more mathematically stable when we're dealing with probabilities than the normal semiring, we convert our data to log-space, do the computations, then convert back.

## Problem Statement

When implementing a CRF model in PyTorch, one of the core building blocks is being able to do the `log-sum-exp` operation over pairs of matrices. Specifically, when we are building our dynamic programming tables, on each timestep `i` we multiply (in log space, add) the potentials with the states from the previous timestep `i-1`, then add (in log space, log-sum-exp) the results together to get the new state. Fortunately, in PyTorch, the `logsumexp` function has already been implemented. Here is the PyTorch version of the function we're trying to optimize, plus the code for benchmarking:

{% highlight python %}
import enum
from typing import Any, Iterable

import click
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor
from tqdm import tqdm

DEFAULT_NFEATS = [2, 4, 8, 16, 32, 64, 128, 256]

## For rendering the SVG text.
plt.rcParams['svg.fonttype'] = 'none'
{% endhighlight %}

Here is our simple implementation of the log-sum-exp function in PyTorch:

{% highlight python %}
def log_bmm(a: Tensor, b: Tensor) -> Tensor:
    """Performs a batch matrix-matrix product of matrices in log-space.
    Args:
        a: tensor with shape (b, n, m)
        b: tensor with shape (b, m, p)
    Returns:
        tensor with shape (b, n, p)
    """

    assert a.ndim == b.ndim == 3
    assert a.size(0) == b.size(0)
    assert a.size(2) == b.size(1)

    bsz, p, m = a.size()
    _, _, n = b.size()
    a = a.unsqueeze(2).expand(bsz, p, n, m)
    b = b.unsqueeze(1).transpose(2, 3).expand(bsz, p, n, m)
    return torch.logsumexp(a + b, dim=-1)
{% endhighlight %}

I'm a big fan of using [click][click-docs] for command line tools, rather than `argparse` - I find that it simplifies building hierarchical tools and looks nicer as code. Here's the benchmarking code:

{% highlight python %}
@click.command()
@click.option('-n', '--nfeats', multiple=True, type=int, default=DEFAULT_NFEATS)
@click.option('-b', '--bsz', type=int, default=8)
@click.option('-t', '--num-trials', type=int, default=10)
def main(nfeats: Iterable[int], bsz: int, num_trials: int) -> None:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    cols = ['nfeat', 'time', 'mem', 'mode']
    df = pd.DataFrame(columns=cols)
    i = 0

    for nfeat in tqdm(nfeats):
        for _ in range(num_trials):
            a = torch.randn(bsz, nfeat, nfeat).cuda()
            b = torch.randn(bsz, nfeat, nfeat).cuda()
            a.requires_grad_(True)
            b.requires_grad_(True)
            
            torch.cuda.reset_max_memory_allocated()
            
            # Runs forward pass.
            start.record()
            o = log_bmm(a, b)
            end.record()
            torch.cuda.synchronize()
            fwd_time = start.elapsed_time(end)
            fwd_mem = torch.cuda.max_memory_allocated()
            df.loc[i] = [nfeat, fwd_time, fwd_mem, 'Forward']
            i += 1

            # Runs backward pass.
            start.record()
            o.sum().backward()
            end.record()
            torch.cuda.synchronize()
            bkwd_time = start.elapsed_time(end)
            bkwd_mem = torch.cuda.max_memory_allocated()
            df.loc[i] = [nfeat, bkwd_time, bkwd_mem, 'Backward']
            i += 1

    fig = sns.catplot(x='nfeat', y='time', data=df, hue='backend', kind='bar', col='mode')
    plt.semilogy()
    fig.savefig('results/time.svg', format='svg')

    fig = sns.catplot(x='nfeat', y='mem', data=df, hue='backend', kind='bar', col='mode')
    plt.semilogy()
    fig.savefig('results/mem.svg', format='svg')

{% endhighlight %}

Lastly, at the end of the file it's important to add some boilerplate to run the script:

{% highlight python %}
if __name__ == '__main__':
    main()
{% endhighlight %}

The naive implementation of this function as implemented gives us a useful baseline. Running the benchmark code above for this function gives us the following memory usage stats (note that the y-axis is log-scaled):

{% include /images/logsumexp/perf/naive_mem.svg %}

Here is the corresponding chart for the runtime:

{% include /images/logsumexp/perf/naive_time.svg %}

### Simple Speed-up

There is a very simple speed-up that we can do on the above function by preserve memory continuity. Note that the reduction part of the forward pass will take place over the last dimension, so we want to make sure the last dimension is contiguous in memory. If we remove the `transpose` part, the forward pass can be performed much faster.

{% highlight python %}
def log_bmm(a: Tensor, b: Tensor) -> Tensor:
    assert a.ndim == b.ndim == 3
    assert a.size(0) == b.size(0)
    assert a.size(1) == b.size(1)

    bsz, p, m = a.size()
    _, n, _ = b.size()
    a = a.unsqueeze(2).expand(bsz, p, n, m)
    b = b.unsqueeze(2).expand(bsz, p, n, m)
    return torch.logsumexp(a + b, dim=-1)
{% endhighlight %}

The memory usage for this function is identical to the corresponding function:

{% include /images/logsumexp/perf/naive_better_mem.svg %}

However, the runtime for the forward and backward passes is slightly faster:

{% include /images/logsumexp/perf/naive_better_time.svg %}

This is a useful lesson: **where possible, avoid transposes**. Note that the above function will return a different result from our canonical implementation, so it is the caller's responsibility to make sure the inputs are correct.

### CUDA Implementation

Let's write a CUDA implementation of the above function, to see if we can improve the performance.

We can write the `log_bmm` function as a matrix-matrix operation (the batch part can be added trivially in the CUDA implementation). For a regular batch matrix multiplication function, we expect as our inputs two matrices with elements $a_{i, j}$ and $b_{i, j}$. We will output a matrix with elements $o_{i, j}$$, which is defined as the following:

$$o_{i, j} = \sum_k a_{i, k} b_{k, j}$$

The log-space version of this function is instead:

$$o_{i, j} = \log \sum_k \exp(a_{i, k} + b_{k, j})$$

Note that, to make this function mathematically stable, we use the `logsumexp` trick above, rather than naively summing over the exponents.

We can differentiate the above function with respect to each $a_{i, k}$ and $b_{k, j}$ giving:

$$
\begin{aligned}
\frac{\delta o_{i, j}}{\delta a_{i, k}} = \frac{\delta o_{i, j}}{\delta b_{k, j}} = & \frac{\exp(a_{i, k} + b_{k, j})}{\sum_{k'} \exp(a_{i, k'} + b_{k', j})} \\
= & \frac{\exp(a_{i, k} + b_{k, j})}{\exp(o_{i, j})} \\
= & \exp(a_{i, k} + b_{k, j} - o_{i, j})
\end{aligned}
$$

This means that gradients of the loss function with respect to $a_{i, k}$ can be written as the accumulation of all of the gradients $\frac{\delta L}{\delta o_{i, j}}$:

$$\frac{\delta L}{\delta a_{i, k}} = \sum_j \exp(a_{i, k} + b_{k, j} - o_{i, j}) \frac{\delta L}{\delta o_{i, j}}$$

Similarly, the gradient with respect to $b_{k, j}$ can be written as:

$$\frac{\delta L}{\delta b_{j, k}} = \sum_i \exp(a_{i, k} + b_{k, j} - o_{i, j}) \frac{\delta L}{\delta o_{i, j}}$$

For the CUDA implementation below, I'm using some of the constants defined in [this post][cuda-post]. Here's the relevant headers and aliases:

{% highlight cuda %}
#include "defs.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

// Type alias for packed tensor arguments.
template <typename scalar_t, int dims>
using PackedAccessor =
    torch::PackedTensorAccessor32<scalar_t, dims, torch::RestrictPtrTraits>;
{% endhighlight %}

Let's write the forward pass of the algorithm. Here are some notes about this implementation:

- In log space, zero is represented as negative infinity. For practical purposes we can just choose a large negative number.
- We first find the maximum value over each element, then add together all the elements minus this maximum. This is mathematically identical to the formulation above (although, as we'll see below, this can be improved on).
- We can fiddle with the number of threads. For the block size, we use the identity `(x + y - 1) / y` to do integer division `x / y` rounding up, to ensure that we are allocating a sufficient number of blocks.

Let's see the code:

{% highlight cuda %}
template <typename scalar_t>
__global__ void logbmm_fp_kernel(const PackedAccessor<scalar_t, 3> a,
                                 const PackedAccessor<scalar_t, 3> b,
                                 PackedAccessor<scalar_t, 3> out,
                                 const int in_size, const int a_size,
                                 const int b_size) {
  const int n = blockIdx.z, row = threadIdx.x + blockIdx.x * blockDim.x,
            col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < b_size) {
    scalar_t val = 0.0;
    scalar_t m = -1e9; // Large negative number.
    for (int i = 0; i < in_size; i++) {
      scalar_t v = a[n][row][i] + b[n][i][col];
      if (v > m) {
        m = v;
      }
    }
    for (int i = 0; i < in_size; i++) {
      scalar_t v = a[n][row][i] + b[n][i][col];
      val += exp(v - m);
    }
    out[n][row][col] = log(val) + m;
  }
}

torch::Tensor forward_pass(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_DIMS(a, 3, "First tensor");
  CHECK_DIMS(b, 3, "Second tensor");
  CHECK_DIM(b, a.size(0), "Second tensor", 0);
  CHECK_DIM(b, a.size(2), "Second tensor", 1);

  const long bsz = a.size(0), in_size = a.size(2), a_size = a.size(1),
             b_size = b.size(2);

  const size_t nthreads = 32;
  const dim3 blocks(a_size / nthreads + 1, b_size / nthreads + 1, bsz);
  const dim3 threads_per_block(nthreads, nthreads, 1);

  // Create output placeholder tensor on the same device as the first input.
  auto options = torch::TensorOptions().dtype(a.dtype()).device(
      torch::kCUDA, a.device().index());
  auto out = torch::empty({bsz, a_size, b_size}, options);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      a.type(), "logbmm_fp", ([&] {
        logbmm_fp_kernel<scalar_t><<<blocks, threads_per_block>>>(
            a.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            b.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            in_size, a_size, b_size);
      }));

  return out;
}
{% endhighlight %}

Great! Let's see what the backward pass looks like. Note that we have to backpropagate to both input tensors, so we need two kernels running in parallel streams. While there is more code, it largely uses the same general idea as the forward pass kernel.

{% highlight cuda %}
template <typename scalar_t>
__global__ void logbmm_bp_kernel_a(PackedAccessor<scalar_t, 3> grad_a,
                                   const PackedAccessor<scalar_t, 3> a,
                                   const PackedAccessor<scalar_t, 3> b,
                                   const PackedAccessor<scalar_t, 3> part,
                                   const PackedAccessor<scalar_t, 3> grad_out,
                                   const int in_size, const int a_size,
                                   const int b_size) {
  const int n = blockIdx.z, row = threadIdx.x + blockIdx.x * blockDim.x,
            col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < in_size) {
    scalar_t val = 0.0;
    for (int k = 0; k < b_size; k++) {
      scalar_t v = a[n][row][col] + b[n][col][k] - part[n][row][k];
      val += exp(v) * grad_out[n][row][k];
    }
    grad_a[n][row][col] = val;
  }
}

template <typename scalar_t>
__global__ void logbmm_bp_kernel_b(PackedAccessor<scalar_t, 3> grad_b,
                                   const PackedAccessor<scalar_t, 3> a,
                                   const PackedAccessor<scalar_t, 3> b,
                                   const PackedAccessor<scalar_t, 3> part,
                                   const PackedAccessor<scalar_t, 3> grad_out,
                                   const int in_size, const int a_size,
                                   const int b_size) {
  const int n = blockIdx.z, row = threadIdx.x + blockIdx.x * blockDim.x,
            col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < in_size && col < b_size) {
    scalar_t val = 0.0;
    for (int k = 0; k < a_size; k++) {
      scalar_t v = a[n][k][row] + b[n][row][col] - part[n][k][col];
      val += exp(v) * grad_out[n][k][col];
    }
    grad_b[n][row][col] = val;
  }
}

std::vector<torch::Tensor> backward_pass(torch::Tensor a, torch::Tensor b,
                                         torch::Tensor grad_out,
                                         torch::Tensor part) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(part);
  CHECK_DIMS(a, 3, "First tensor");
  CHECK_DIMS(b, 3, "Second tensor");
  CHECK_DIMS(grad_out, 3, "Gradients");
  CHECK_DIMS(part, 3, "Output");
  CHECK_DIM(b, a.size(0), "Second tensor", 0);
  CHECK_DIM(b, a.size(2), "Second tensor", 1);
  CHECK_DIM(grad_out, a.size(0), "Gradients", 0);
  CHECK_DIM(grad_out, a.size(1), "Gradients", 1);
  CHECK_DIM(grad_out, b.size(2), "Gradients", 2);
  CHECK_DIM(part, a.size(0), "Output", 0);
  CHECK_DIM(part, a.size(1), "Output", 1);
  CHECK_DIM(part, b.size(2), "Output", 2);

  const long bsz = a.size(0), in_size = a.size(2), a_size = a.size(1),
             b_size = b.size(2);

  const size_t nthreads = 32;
  const dim3 blocks_a(a_size / nthreads + 1, in_size / nthreads + 1, bsz);
  const dim3 blocks_b(in_size / nthreads + 1, b_size / nthreads + 1, bsz);
  const dim3 threads_per_block(nthreads, nthreads, 1);

  // Placeholder tensors for output gradients.
  auto grad_a = torch::empty_like(a);
  auto grad_b = torch::empty_like(b);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      a.type(), "logbmm_bp", ([&] {
        // Creates streams for both gradient kernels.
        cudaStream_t a_stream, b_stream;
        cudaStreamCreate(&a_stream);
        cudaStreamCreate(&b_stream);

        logbmm_bp_kernel_a<
            scalar_t><<<blocks_a, threads_per_block, 0, a_stream>>>(
            grad_a.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            a.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            b.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            part.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            in_size, a_size, b_size);
        logbmm_bp_kernel_b<
            scalar_t><<<blocks_b, threads_per_block, 0, b_stream>>>(
            grad_b.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            a.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            b.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            part.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            in_size, a_size, b_size);

        // Synchronizes streams once both kernels are finished.
        cudaStreamSynchronize(a_stream);
        cudaStreamSynchronize(b_stream);
        cudaStreamDestroy(a_stream);
        cudaStreamDestroy(b_stream);
      }));

  return {grad_a, grad_b};
}
{% endhighlight %}

Lastly, we'll add some boilerplate for [pybind11][pybind11-docs] to be able to access our functions from the Python side. I think it usually makes sense to have the forward and backward passes in their own submodule, for coherence.

{% highlight cuda %}
void init_py(pybind11::module &m) {
  pybind11::module sub_m =
      m.def_submodule("forward_backward", "Batch log-sum-exp function");

  sub_m.def("forward_pass", &forward_pass, "Forward pass function");
  sub_m.def("backward_pass", &backward_pass, "Gradient propagation function");
}
{% endhighlight %}

There is some additional boilerplate that is missing from the above code. See [here][cuda-extension-writeup] for a complete tutorial on how to compile this CUDA kernel for your extension. The general idea for incorporating this as a PyTorch function is:

{% highlight python %}
class _LogBMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        out = cuda.logsumexp.forward_pass(a, b)
        ctx.save_for_backward(a, b, out)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b, out = ctx.saved_tensors
        grad_a, grad_b = cuda.logsumexp.backward_pass(
            a, b, grad_output.contiguous(), out)
        return grad_a, grad_b


def log_bmm(a: Tensor, b: Tensor) -> Tensor:
    return _LogBMM.apply(a, b)
{% endhighlight %}

Plugging this function into our performance benchmarking script gives the following chart for memory usage:

{% include /images/logsumexp/perf/cuda_first_mem.svg %}

Here is the corresponding chart for the runtime:

{% include /images/logsumexp/perf/cuda_first_time.svg %}

This is a quite significant improvement in memory usage and runtime! It turns out that we can save a substantial amount of memory by writing a pure CUDA implementation.

### Improving the CUDA Implementation

There are a few things we can do to improve on this baseline CUDA implementation.

### Reduce Memory Accesses

We can cut the number of memory accesses in the forward function in half by performing element-wise `logsumexp` instead of getting the global max. We can do that with the following function:

{% highlight cuda %}
template <typename scalar_t>
__device__ scalar_t logsumexp(scalar_t a, scalar_t b) {
  const scalar_t m = max(a, b);
  return log(exp(a - m) + exp(b - m)) + m;
}

template <typename scalar_t>
__global__ void logbmm_fp_kernel(const PackedAccessor<scalar_t, 3> a,
                                 const PackedAccessor<scalar_t, 3> b,
                                 PackedAccessor<scalar_t, 3> out,
                                 const long in_size, const long a_size,
                                 const long b_size) {
  const long n = blockIdx.z, row = threadIdx.x + blockIdx.x * blockDim.x,
             col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < b_size) {
    scalar_t val = log(0.0);
    for (long i = 0; i < in_size; i++) {
      val = logsumexp(val, a[n][row][i] + b[n][i][col]);
    }
    out[n][row][col] = val;
  }
}
{% endhighlight %}

This gives the following graph for memory usage:

{% include /images/logsumexp/perf/cuda_second_mem.svg %}

Here is the corresponding chart for the runtime:

{% include /images/logsumexp/perf/cuda_second_time.svg %}

### Tree Reduction

There has been a lot of work on how to optimize reduction operations on GPUs, including a great [tutorial][optimizing-parallel-reduction] by NVIDIA. There are a lot of tricks involved in doing this efficiently (for more info, see that post), but the basic idea is that we want to avoid doing the reduction operations serially, like in the image below.

{% include /images/logsumexp/linear_reduction.svg %}

Instead, when the reduction operation is **associative** and **commutative** (which, fortunately, is the case for all semirings, not just the one in question), we can perform them with $O(\log(N))$ parallel steps, as in the tree below.

{% include /images/logsumexp/tree_reduction.svg %}

These performance boosts don't really apply for reduce operations which are relatively small (and by "small", I mean less than ~1000 dimensions).

[prev-post]: {% post_url 2020-04-07-hmms-crfs %}
[cuda-post]: {% post_url 2020-05-06-torch-cuda-tricks %}
[optimizing-parallel-reduction]: <https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf>
[genbmm]: <https://github.com/harvardnlp/genbmm>
[log-semiring]: <https://en.wikipedia.org/wiki/Log_semiring>
[semiring-paper]: <https://www.aclweb.org/anthology/J99-4004/>
[double-precision]: <https://en.wikipedia.org/wiki/Double-precision_floating-point_format>
[cuda-extension-writeup]: <https://pytorch.org/tutorials/advanced/cpp_extension.html>
[click-docs]: <https://click.palletsprojects.com/en/7.x/>
[pybind11-docs]: <https://pybind11.readthedocs.io/en/stable/basics.html>
