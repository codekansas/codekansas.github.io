---
layout: post
title: "Investigating the RWKV Language Model"
category: 🔬
excerpt: >
  In-depth explanation of the math behind the RWKV model, with PyTorch
  implementations, plus a discussion of numerical stability.
---

Lately I've found myself spending a lot of time messing around with the [RWKV model][rwkv-model]. It's a cool model, but it's a bit more involved to wrap my head around than vanilla transfomers or their variants. I found [this blog][blog-post] to be quite helpful for understanding the mechanics, as well as the corresponding simplified inference implementation [here][minimal-inference].

In this post, I write out the equations for the core WKV part of the RWKV model, and derive two numerically stable versions - one following the official implementation, another by transforming the state variables to log space - and provide implementations for each in PyTorch. Additionally, I derive the gradients for the log-space version, and provide Triton kernels for training a numerically-stable RWKV model.

{% katexmm %}

## Math

> This section covers the basic math concepts for the WKV operator. If you're already familiar with the math, you can skip to the [PyTorch implementation](#pytorch-implementation) or the [next section](#numerical-stability), which extends the vanilla computation to be numerically stable. Additionally, the gradients for this computation are derived in a [further section](#gradients).

The main "attention" component in the RWKV model is the WKV computation. The equation for this is:

$$
\text{wkv}_i = \frac{ e^{u+k_i} v_i + \sum_{j=1}^{i-1} e^{-(i-1-j)w+k_j} v_j}{e^{u+k_i} + \sum_{j=1}^{i-1} e^{-(i-1-j)w+k_j} }
$$

We can rewrite this using two recursive state variables for the numerator and denominator, which we'll call $\alpha_i$ and $\beta_i$ respectively:

$$
\begin{aligned}
\alpha_i & = \sum_{j=1}^i e^{-(i-j)w+k_j} v_j \\
& = e^{w} \alpha_{i-1} + e^{k_i} v_i \\[1em]
\beta_i & = \sum_{j=1}^i e^{-(i-j)w+k_j} \\
& = e^{w} \beta_{i - 1} + e^{k_i} \\
\end{aligned}
$$

We can then rewrite the WKV computation as:

$$\text{wkv}_i = \frac{ e^{u+k_i} v_i + \alpha_{i - 1} }{ e^{u+k_i} + \beta_{i - 1} }$$

### PyTorch Implementation

A pure-PyTorch implementation of the above WKV computation is below:

```python
def wkv_vanilla_forward(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    bsz, tsz, chans = k.shape
    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 2, 1, chans)

    alpha, beta = state[:, :, -1].chunk(2, dim=1)  # (B, 1, D), (B, 1, D)

    ew = torch.exp(w)

    wkvs = []
    alphas = [alpha]
    betas = [beta]

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        euk = torch.exp(u + kt)
        wkv = (alpha + euk * vt) / (beta + euk)
        wkvs.append(wkv)

        ek = torch.exp(kt)
        alpha = ew * alpha + ek * vt
        beta = ew * beta + ek

        alphas.append(alpha)
        betas.append(beta)

    alpha = torch.stack(alphas, dim=2)
    beta = torch.stack(betas, dim=2)

    return torch.cat(wkvs, 1), torch.cat((alpha, beta), dim=1)
```

## Numerical Stability

> This section extends the vanilla WKV computation discussed [above](#math) to be numerically stable by adding a scaling factor to the exponent. The [PyTorch implementation](#pytorch-implementation-1) might be easier for some readers to follow. The variable names in the code follow the same convention as the math equations. This section explains the numerical stability approach used in the official implementation. The [next section](#log-space-operations) explains an alternative approach that uses log-space state variables to achieve numerical stability instead.

With traditional RNNs, there's a common problem of exploding or vanishing gradients, if the determinant of Jacobian of the hidden state variable is not close to 1. This is because, for long sequences, the same matrix is applied many times, exacerbating any deviation from 1.

With the RWKV model, if the values of $w$ and $k_i$ are large, the exponent can grow beyond the numerical limits of our floating point type. We can solve this using another state variable, which we'll call $\epsilon_i$:

$$
\begin{aligned}
\alpha_i' & = e^{-\epsilon_i} \alpha_i \\
& = e^{w - \epsilon_i} \alpha_{i - 1} + e^{k_i - \epsilon_i} v_i \\
& = e^{w + \epsilon_{i - 1} - \epsilon_i} \alpha_{i - 1}' + e^{k_i - \epsilon_i} v_i \\[1em]
\beta_i' & = e^{-\epsilon_i} \beta_i \\
& = e^{w - \epsilon_i} \beta_{i - 1} + e^{k_i - \epsilon_i} \\
& = e^{w + \epsilon_{i - 1} - \epsilon_i} \beta_{i - 1}' + e^{k_i - \epsilon_i} \\
\end{aligned}
$$

This allows us to rewrite the WKV computation as:

$$
\begin{aligned}
\text{wkv}_i & = \frac{ e^{u+k_i} v_i + e^{\epsilon_{i - 1}}\alpha_{i - 1}' }{ e^{u+k_i} + e^{\epsilon_{i - 1}}\beta_{i - 1}' } \\
& = \frac{ e^{u+k_i-\epsilon_{i - 1}} v_i + \alpha_{i - 1}' }{ e^{u+k_i-\epsilon_{i - 1}} + \beta_{i - 1}' } \\
\end{aligned}
$$

We can add an additional scaling factor $\tau_i$ and multiply by $\frac{e^{-\tau_i}}{e^{-\tau_i}}$ to get:

$$\text{wkv}_i = \frac{ e^{u+k_i-\tau_i} v_i + e^{\epsilon_{i - 1}-\tau_i}\alpha_{i - 1}' }{ e^{u+k_i-\tau_i} + e^{\epsilon_{i - 1}-\tau_i}\beta_{i - 1}' }$$

The value of $\epsilon_i$ is arbitrary, and since we want to keep $e^{w + \epsilon_{i - 1} - \epsilon_i}$ and $e^{k_i - \epsilon_i}$ less than 1 to prevent it growing really large, we can set it as:

$$\epsilon_{i} = \max(w + \epsilon_{i - 1}, k_i)$$

Then, to keep $e^{u + k_i - \tau_i}$ and $e^{\epsilon_{i - 1} - \tau_i}$ less than 1, we can set $\tau_i$ as:

$$\tau_i = \max(u + k_i, \epsilon_{i - 1})$$

So ultimately we have three state variables:

- $\alpha_i'$, which is the numerator of the WKV computation
- $\beta_i'$, which is the denominator of the WKV computation
- $\epsilon_i$, which is used to maintain numerical stability

$\alpha_i'$ and $\beta_i'$ are accumulated over time, while $\epsilon_i$ is just passed to the subsequent step (in other words, $\text{wkv}_i$ depends on $\epsilon_{i-1}$, but $\epsilon_i$ doesn't).

### PyTorch Implementation

The PyTorch implementation of the more stable form of the WKV computation follows fairly directly from the equations above.

```python
def wkv_with_eps_forward(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    assert w.dim() == u.dim() == 1
    assert k.dim() == v.dim() == 3
    assert state.dim() == 4

    alpha, beta, eps = state[:, :, -1].chunk(3, dim=1)  # (B, 1, D), (B, 1, D), (B, 1, D)

    _, tsz, _ = k.shape

    wkvs = []
    alphas = [alpha]
    betas = [beta]
    epss = [eps]

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        ukt = u + kt
        tau = torch.maximum(ukt, eps)
        e1 = torch.exp(eps - tau)
        e2 = torch.exp(ukt - tau)
        wkv = (e1 * alpha + e2 * vt) / (e1 * beta + e2)
        wkvs.append(wkv)

        ww = eps + w
        eps = torch.maximum(ww, kt)
        e1 = torch.exp(ww - eps)
        e2 = torch.exp(kt - eps)
        alpha = e1 * alpha + e2 * vt
        beta = e1 * beta + e2

        alphas.append(alpha)
        betas.append(beta)
        epss.append(eps)

    alpha = torch.stack(alphas, dim=2)
    beta = torch.stack(betas, dim=2)
    eps = torch.stack(epss, dim=2)

    return torch.cat(wkvs, 1), torch.cat((alpha, beta, eps), dim=1)
```

## Log-Space Operations

> This section provides an alternative approach to achieving numerical stability in the WKV computation to the approach described [above](#numerical-stability), by using log-space operations. This approach should be familiar to anyone who has dealt with log-domain Viterbi algorithms or graphical models, and is included here mainly as a point of comparison with the approach described above. For readers who are more comfortable reading code than equations, you can skip directly to the [PyTorch implementation](#pytorch-implementation-2).

The prevalence of exponentials in the WKV computation suggests that it might be a good idea to perform some operations in log-space. For example, if we consider the vanilla $\alpha_i$ and $\beta_i$ equations:

$$
\begin{aligned}
\alpha_i & = e^{w} \alpha_{i-1} + e^{k_i} v_i \\
\beta_i & = e^{w} \beta_{i - 1} + e^{k_i} \\
\end{aligned}
$$

If we were guaranteed that the values of $\alpha_{i-1}$, $\beta_{i-1}$ and $v_i$ were always positive, we could take advantage of the [log-sum-exp][log-sum-exp] trick for computing the log-sum of exponentials in a numerically stable way:

$$LSE(x, y) = \log(e^x + e^y) = \max(x, y) + \log(1 + e^{-|x - y|})$$

Re-writing the equations in log-space, we get:

$$
\begin{aligned}
\log \alpha_i & = \log(e^{w} \alpha_{i-1} + e^{k_i} v_i) \\
& = LSE(w + \log \alpha_{i-1}, k_i + \log v_i) \\[1em]
\log \beta_i & = \log(e^{w} \beta_{i - 1} + e^{k_i}) \\
& = LSE(w + \log \beta_{i - 1}, k_i) \\
\end{aligned}
$$

Revisiting our WKV equation:

$$\text{wkv}_i = \frac{ e^{u+k_i} v_i + \alpha_{i - 1} }{ e^{u+k_i} + \beta_{i - 1} }$$

We can re-write this in log-space as:

$$
\begin{aligned}
\log \text{wkv}_i & = \log \left( \frac{ e^{u+k_i} v_i + \alpha_{i - 1} }{ e^{u+k_i} + \beta_{i - 1} } \right) \\
& = \log(e^{u+k_i} v_i + \alpha_{i - 1}) - \log(e^{u+k_i} + \beta_{i - 1}) \\
& = LSE(u + k_i + \log v_i, \log \alpha_{i - 1}) - LSE(u + k_i, \log \beta_{i - 1}) \\
\end{aligned}
$$

The advantage here is that we no longer need to store $\epsilon_i$ between steps.

### Reparametrization

In order to avoid trying to take the log of a negative value, we need to make sure that $v_i$ is strictly positive. We can do so by reparametrizing $v_i$ as the sum of its positive and negative parts:

$$
\begin{aligned}
v_i^- & = -\min(v_i, 0) + \epsilon \\
v_i^+ & = \max(v_i, 0) + \epsilon \\[1em]
v_i & = v_i^+ - v_i^- \\
\end{aligned}
$$

Note that the $\epsilon$ in this equation is a small value added for numerical stability, not the $\epsilon_i$ from earlier. This reparametrization ensures that the values of $v_i^+$ and $v_i^-$ will always be in the range $[\epsilon, \infty)$ and therefore will have a non-imaginary log value.

We can take advantage of this fact to rewrite our equation for $\alpha_i$:

$$
\begin{aligned}
\alpha_i & = \sum_{j=1}^i e^{-(i-j)w+k_j} (v_j^+ - v_j^-) \\
& = \sum_{j=1}^i e^{-(i-j)w+k_j} v_j^+ - \sum_{j=1}^i e^{-(i-j)w+k_j} v_j^- \\
\end{aligned}
$$

Separating out $\alpha_i = \alpha_i^+ - \alpha_i^-$, we get:

$$
\begin{aligned}
\alpha_i^+ & = e^w \alpha_{i - 1}^+ + e^{k_j} v_j^+ \\
\alpha_i^- & = e^w \alpha_{i - 1}^- + e^{k_j} v_j^- \\
\end{aligned}
$$

Finally, we can incorporate $v_i^+$, $v_i^-$, $\alpha_i^+$ and $\alpha_i^-$ into our log-space equations for $\text{wkv}_i$:

$$
\begin{aligned}
\text{wkv}_i = \frac{ e^{u+k_i} (v_i^+ - v_i^-) + (\alpha_{i - 1}^+ - \alpha_{i - 1}^-) }{ e^{u+k_i} + \beta_{i - 1} } \\
\text{wkv}_i = \frac{ e^{u+k_i} v_i^+ + \alpha_{i - 1}^+ }{ e^{u+k_i} + \beta_{i - 1} } - \frac{ e^{u+k_i} v_i^- + \alpha_{i - 1}^- }{ e^{u+k_i} + \beta_{i - 1} }\\
\end{aligned}
$$

Separating out $\text{wkv}_i = \text{wkv}_i^+ - \text{wkv}_i^-$, we get:

$$
\begin{aligned}
\text{wkv}_i^+ & = \frac{ e^{u+k_i} v_i^+ + \alpha_{i - 1}^+ }{ e^{u+k_i} + \beta_{i - 1} } \\
\log \text{wkv}_i^+ & = LSE(u + k_i + \log v_i^+, \log \alpha_{i - 1}^+) - LSE(u + k_i, \log \beta_{i - 1}) \\[1em]
\text{wkv}_i^- & = \frac{ e^{u+k_i} v_i^- + \alpha_{i - 1}^- }{ e^{u+k_i} + \beta_{i - 1} } \\
\log \text{wkv}_i^- & = LSE(u + k_i + \log v_i^-, \log \alpha_{i - 1}^-) - LSE(u + k_i, \log \beta_{i - 1}) \\
\end{aligned}
$$

Note that while we no longer need to use $\epsilon_i$ as a state variable, we now need to carry $\alpha_i^+$ and $\alpha_i^-$.

### PyTorch Implementation

We can implement the above equations in PyTorch as follows:

```python
def wkv_log_space_forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    assert w.dim() == u.dim() == 1
    assert k.dim() == v.dim() == state.dim()

    log_alpha_plus, log_alpha_minus, log_beta = state.chunk(3, dim=1)

    _, tsz, _ = k.shape

    wkvs = []

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        v_plus = torch.clamp(vt, min=0) + eps
        v_minus = torch.clamp(-vt, min=0) + eps
        log_v_plus = torch.log(v_plus)
        log_v_minus = torch.log(v_minus)

        log_wkv_plus = torch.logaddexp(u + kt + log_v_plus, log_alpha_plus) - torch.logaddexp(u + kt, log_beta)
        log_wkv_minus = torch.logaddexp(u + kt + log_v_minus, log_alpha_minus) - torch.logaddexp(u + kt, log_beta)

        wkv = torch.exp(log_wkv_plus) - torch.exp(log_wkv_minus)
        wkvs.append(wkv)

        log_alpha_plus = torch.logaddexp(w + log_alpha_plus, kt + log_v_plus)
        log_alpha_minus = torch.logaddexp(w + log_alpha_minus, kt + log_v_minus)
        log_beta = torch.logaddexp(w + log_beta, kt)

    return torch.cat(wkvs, 1), torch.cat((log_alpha_plus, log_alpha_minus, log_beta), dim=1)
```

## Gradients

> This section implements a manual backward pass for the vanilla WKV computation [described above](#math), first derived from the equations above, then implemented in PyTorch.

When we are actually implementing this in PyTorch, we will want to write an optimized kernel for performing the WKV computation. The downside is that it means we can't use autograd to compute the gradients for us, so we need to derive equations for the gradients.

Revisiting our original equation for the WKV computation:

$$\text{wkv}_i = \frac{ e^{u+k_i} v_i + \alpha_{i - 1} }{ e^{u+k_i} + \beta_{i - 1} }$$

The partial derivatives of the WKV computation with respect to $u$, $k$, and $v$ are as follows:

$$
\begin{aligned}
\frac{\partial \text{wkv}_i}{\partial u} & = \frac{ e^{u + k_i} v_i}{e^{u + k_i} + \beta_{i - 1}} - \frac{ e^{u + k_i} (e^{u + k_i} v_i + \alpha_{i - 1})}{(e^{u + k_i} + \beta_{i - 1})^2} \\
& = \frac{ e^{u + k_i} (\beta_{i - 1} v_i - \alpha_{i - 1})}{(\beta_{i - 1} + e^{u + k_i})^2} \\[1.5em]
\frac{\partial \text{wkv}_i}{\partial k_i} & = \frac{ e^{u + k_i} v_i}{e^{u + k_i} + \beta_{i - 1}} - \frac{ e^{u + k_i} (e^{u + k_i} v_i + \alpha_{i - 1})}{(e^{u + k_i} + \beta_{i - 1})^2} \\
& = \frac{ e^{u + k_i} (\beta_{i - 1} v_i - \alpha_{i - 1})}{(\beta_{i - 1} + e^{u + k_i})^2} \\[1.5em]
\frac{\partial \text{wkv}_i}{\partial v_i} & = \frac{ e^{u + k_i}}{e^{u + k_i} + \beta_{i-1}} \\[1.5em]
\frac{\partial \text{wkv}_i}{\partial \alpha_{i-1}} & = \frac{1}{e^{u + k_i} + \beta_{i-1}} \\[1.5em]
\frac{\partial \text{wkv}_i}{\partial \beta_{i-1}} & = -\frac{v e^{u + k_i} + \alpha_{i-1}}{(e^{u + k_i} + \beta_{i-1})^2} \\
\end{aligned}
$$

We also need to compute the partial derivatives of $\alpha_i$ and $\beta_i$. Fortunately these are comparatively simple. Revisiting our original equations:

$$
\begin{aligned}
\alpha_i & = e^{w} \alpha_{i - 1} + e^{k_i} v_i \\
\beta_i & = e^{w} \beta_{i - 1} + e^{k_i} \\
\end{aligned}
$$

For $\alpha_i$ we have:

$$
\begin{aligned}
\frac{\partial \alpha_i}{\partial w} & = e^{w} \alpha_{i - 1} \;
& \frac{\partial \alpha_i}{\partial k_i} & = e^{k_i} v_i \\[1.5em]
\frac{\partial \alpha_i}{\partial v_i} & = e^{k_i} \;
& \frac{\partial \alpha_i}{\partial \alpha_{i-1}} & = e^{w}
\end{aligned}
$$

For $\beta_i$ we have:

$$
\frac{\partial \beta_i}{\partial w} = e^{w} \beta_{i - 1} \quad
\frac{\partial \beta_i}{\partial k_i} = e^{k_i} \quad
\frac{\partial \beta_i}{\partial \beta_{i-1}} = e^{w}
$$

### PyTorch Implementation

We can manually implement the above equations in PyTorch. This implementation is more academic than practical, since it's a straightforward function to implement as a CUDA or Triton kernel, but sometimes it is easier to read code than equations (also, and perhaps more importantly, it lets us write unit tests to make sure the equations are correct).

```python
def wkv_vanilla_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape
    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 2, tsz + 1, chans)
    assert grad_wkv.shape == (bsz, tsz, chans)
    assert grad_state.shape == (bsz, 2, 1, chans)

    alpha, beta = state.chunk(2, dim=1)  # (B, 1, T + 1, D), (B, 1, T + 1, D)
    grad_alpha, grad_beta = grad_state[:, :, 0].chunk(2, dim=1)  # (B, 1, D), (B, 1, D)

    ew = torch.exp(w)

    grad_w = torch.zeros_like(w)
    grad_u = torch.zeros_like(u)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    for t in reversed(range(tsz)):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        euk = torch.exp(u + kt)
        ek = torch.exp(kt)

        alpha_prev, beta_prev = alpha[:, :, t], beta[:, :, t]

        grad_wkvt = grad_wkv[:, t : t + 1]

        # Backpropagates wkv gradients.
        grad_uk = grad_wkvt * euk * (beta_prev * vt - alpha_prev) / (beta_prev + euk) ** 2
        grad_u += grad_uk.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_uk
        grad_v[:, t : t + 1] += grad_wkvt * euk / (beta_prev + euk)

        # Backpropagate alpha gradients.
        grad_w += (grad_alpha * ew * alpha_prev).flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_alpha * ek * vt
        grad_v[:, t : t + 1] += grad_alpha * ek

        # Backpropagate beta gradients.
        grad_w += (grad_beta * ew * beta_prev).flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_beta * ek

        # Compute gradients for alpha and beta.
        grad_alpha = grad_alpha * ew + grad_wkvt / (beta_prev + euk)
        grad_beta = grad_beta * ew - grad_wkvt * (euk * vt + alpha_prev) / (beta_prev + euk) ** 2

    return grad_w, grad_u, grad_k, grad_v, torch.stack((grad_alpha, grad_beta), dim=1)
```

## Log-Space Gradients

In our log-space implementation of the WKV computation, we have:

$$\text{wkv}_i = e^{\log \text{wkv}_i^+} - e^{\log \text{wkv}_i^-}$$

This gives us the following partial derivatives:

$$
\begin{aligned}
\frac{\partial \text{wkv}_i}{\partial \log \text{wkv}_i^+} & = e^{\log \text{wkv}_i^+} & = \text{wkv}_i^+ \\[1.5em]
\frac{\partial \text{wkv}_i}{\partial \log \text{wkv}_i^-} & = -e^{\log \text{wkv}_i^-} & = -\text{wkv}_i^-
\end{aligned}
$$

Next, we need to compute the gradients of $\log \text{wkv}_i^+$ and $\log \text{wkv}_i^-$ with respect to each of our inputs. We have:

$$
\begin{aligned}
\log \text{wkv}_i^+ & = LSE(u + k_i + \log v_i^+, \log \alpha_{i - 1}^+) - LSE(u + k_i, \log \beta_{i - 1}) \\[1em]
\log \text{wkv}_i^- & = LSE(u + k_i + \log v_i^-, \log \alpha_{i - 1}^-) - LSE(u + k_i, \log \beta_{i - 1}) \\
\end{aligned}
$$

The gradients of the log-sum-exp function are given by:

$$\frac{\partial LSE(a, b)}{\partial a} = \frac{e^a}{e^a + e^b} \quad \frac{\partial LSE(a, b)}{\partial b} = \frac{e^b}{e^a + e^b}$$

We can use this to find the partial derivatives:

$$
\begin{aligned}
\frac{\partial y}{\partial u} & = -\frac{e^{k + u}}{b + e^{k + u}} + \frac{e^{k + u} v}{a + e^{k + u} v} \\
\frac{\partial y}{\partial k} & = -\frac{e^{k + u}}{b + e^{k + u}} + \frac{e^{k + u} v}{a + e^{k + u} v} \\
\frac{\partial y}{\partial v} & = \frac{e^{k + u}}{a + e^{k + u} v} \\
\frac{\partial y}{\partial a} & = \frac{1}{a + e^{k + u} v} \\
\frac{\partial y}{\partial b} & = -\frac{1}{b + e^{k + u}} \\
\end{aligned}
$$

{% endkatexmm %}

[rwkv-model]: https://github.com/BlinkDL/RWKV-LM
[blog-post]: https://johanwind.github.io/2023/03/23/rwkv_details.html
[minimal-inference]: https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_in_150_lines.py
[log-sum-exp]: https://en.wikipedia.org/wiki/LogSumExp
