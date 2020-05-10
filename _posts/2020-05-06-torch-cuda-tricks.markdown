---
layout: post
title: "Torch CUDA Extension Tricks"
date: 2020-05-06 12:00:00
category: 🧐
excerpt: Some tricks I found useful for writing CUDA extensions for PyTorch.
math: true
---

This is a tracking document for some things I've found useful when writing CUDA extensions for PyTorch.

# Python

I found it useful to put these at the top of my Python file. `manual_seed` is for reproducability and `set_printoptions` is to make it easier to quickly identify whether or not two numbers match up.

{% highlight python %}
torch.manual_seed(seed)
torch.set_printoptions(precision=6, sci_mode=False)
{% endhighlight %}

# CUDA Debugging

[This answer](https://discuss.pytorch.org/t/whats-the-meaning-of-this-error-how-can-i-debug-when-i-use-gpu/8052/3) suggests the first step for debugging CUDA code is to enable CUDA launch blocking using this at the top of the Python file:

```python
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

However, this didn't work for a weird memory access issue I was having. This [guide](https://nanxiao.me/en/an-empirical-method-of-debugging-illegal-memory-access-bug-in-cuda-programming/) was more helpful.

# C++ Definitions

These are some definitions which I found useful.

{% highlight c++ %}
// Short-hand for getting a packed accessor of a particular type.
#define ACCESSOR(x, n, type)                                                   \
  x.packed_accessor32<type, n, torch::RestrictPtrTraits>()

// Short-hand for getting the CUDA thread index.
#define CUDA_IDX(x) (blockIdx.x * blockDim.x + threadIdx.x)

// Checks that a number is between two other numbers.
#define CHECK_BETWEEN(v, a, b, n)                                              \
  TORCH_CHECK(v >= a && v < b, n, " should be between ", a, " and ", b,        \
              ", got ", v);

// Checks that a tensor has the right dimensionality at some index.
#define CHECK_DIM(x, v, n, d)                                                  \
  TORCH_CHECK(x.size(d) == v, n, " should have size ", v, " in dimension ", d, \
              ", got ", x.size(d))

// Checks that a tensor has the right number of dimensions.
#define CHECK_DIMS(x, d, n)                                                    \
  TORCH_CHECK(x.ndimension() == d, n, " should have ", d, " dimensions, got ", \
              x.ndimension())

// Checks that the tensor is a CUDA tensor.
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

// Checks that the tensor is contiguous.
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Combines the CUDA tensor and contiguousness checks.
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)
{% endhighlight %}

# Additional Resources

Below are some of the resources that I found useful.

- [Illegal Memory Access][illegal-memory-access]: "An empirical method of debugging “illegal memory access” bug in CUDA programming", useful guide for debugging memory issues.
- [CUDA Extension Write-up][cuda-extension-writeup]: Introduces how to get started writing a CUDA extension for PyTorch and walks through a complete code example.
- [Optimizing Parallel Reduction in CUDA][parallel-reduction-slides]: Slides describing how to speed up reductions (useful for any operations on [rings][rings-wiki]).

[cuda-extension-writeup]: https://pytorch.org/tutorials/advanced/cpp_extension.html
[parallel-reduction-slides]: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
[rings-wiki]: https://en.wikipedia.org/wiki/Ring_(mathematics)
[illegal-memory-access]: https://nanxiao.me/en/an-empirical-method-of-debugging-illegal-memory-access-bug-in-cuda-programming/
