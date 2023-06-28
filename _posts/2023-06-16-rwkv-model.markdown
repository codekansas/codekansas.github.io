---
layout: post
title: "RWKV Language Model Math"
tags: [ml, nlp, math]
excerpt: >
  In-depth explanation of the math behind the RWKV model, with PyTorch
  implementations, plus a discussion of numerical stability.
---

Lately I've found myself spending a lot of time messing around with the [RWKV model][rwkv-model]. It's a cool model, but it's a bit more involved to wrap my head around than vanilla transfomers or their variants. I found [this blog][blog-post] to be quite helpful for understanding the mechanics, as well as the corresponding simplified inference implementation [here][minimal-inference].

In this post, I write out the equations for the core WKV part of the RWKV model, and derive two numerically stable versions - one following the official implementation, another by transforming the state variables to log space - and provide implementations for each in PyTorch. Additionally, I derive the gradients for the log-space version, and provide Triton kernels for training a numerically-stable RWKV model.

In most cases, the gradients were verified with Wolfram Alpha, although there may be a typo in the math. The PyTorch implementations are verified by comparing the manual implementation of the backward pass with the autograd implementation. See [this repo][rwkv-repo] to check out the full code and unit tests.

## Math

{% katexmm %}

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
def wkv_vanilla_forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
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
def wkv_with_eps_forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
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

$$
\begin{aligned}
LSE(x, y) & = \log(e^x + e^y) \\
& = \max(x, y) + \log(e^{x - \max(x, y)} + e^{y - \max(x, y)}) \\
& = \max(x, y) + \log(1 + e^{-|x - y|}) \\
\end{aligned}
$$

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

Note that we can renormalize $\alpha_i^+$ and $\alpha_i^-$ by subtracting $\min(\alpha_i^+, \alpha_i^-) - \epsilon$ from both value, which helps prevent the values of $\alpha_i^+$ and $\alpha_i^-$ exploding. Since we are working in the log domain, we should use the subtraction version of the log-sum-exp trick:

$$
\begin{aligned}
\log(e^x - e^y) & = \max(x, y) + \log(e^{x - \max(x, y)} - e^{y - \max(x, y)}) \\
& = \max(x, y) + \log(1 - e^{-|x - y|}) \\
\end{aligned}
$$

This only works since we know that $\alpha_i^+$ and $\alpha_i^-$ are both strictly greater than $\min(\alpha_i^+, \alpha_i^-) - \epsilon$.

Finally, we can incorporate $v_i^+$, $v_i^-$, $\alpha_i^+$ and $\alpha_i^-$ into our log-space equations for $\text{wkv}_i$:

$$
\begin{aligned}
\text{wkv}_i = \frac{ e^{u+k_i} (v_i^+ - v_i^-) + (\alpha_{i - 1}^+ - \alpha_{i - 1}^-) }{ e^{u+k_i} + \beta_{i - 1} } \\
\text{wkv}_i = \frac{ e^{u+k_i} v_i^+ + \alpha_{i - 1}^+ }{ e^{u+k_i} + \beta_{i - 1} } - \frac{ e^{u+k_i} v_i^- + \alpha_{i - 1}^- }{ e^{u+k_i} + \beta_{i - 1} }\\
\end{aligned}
$$

Now we *do* have an equation (or rather, two equations) for $\text{wkv}_i$ with strictly positive values. Separating out $\text{wkv}_i = \text{wkv}_i^+ - \text{wkv}_i^-$, we get:

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
    eps: float = EPS,
    normalize: bool = False,
) -> tuple[Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 3, 1, chans)

    ln_alpha_p, ln_alpha_m, ln_beta = state[:, :, -1].chunk(3, dim=1)

    log_eps = math.log(eps)

    wkvs = []
    ln_alpha_ps = [ln_alpha_p]
    ln_alpha_ms = [ln_alpha_m]
    ln_betas = [ln_beta]

    def logaddexp(a: Tensor, b: Tensor) -> Tensor:
        max_av = torch.maximum(a, b)
        return max_av + torch.log(torch.exp(a - max_av) + torch.exp(b - max_av))

    def logsubexp(a: Tensor, b: Tensor) -> Tensor:
        max_av = torch.maximum(torch.maximum(a, b), torch.full_like(a, log_eps))
        return max_av + torch.log(torch.exp(a - max_av) - torch.exp(b - max_av))

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        vt_p, vt_m = torch.clamp_min(vt, 0) + eps, torch.clamp_min(-vt, 0) + eps
        ln_v_p, ln_v_m = torch.log(vt_p), torch.log(vt_m)

        if normalize:
            ln_alpha_pm = torch.minimum(ln_alpha_p, ln_alpha_m) - eps
            ln_alpha_p = logsubexp(ln_alpha_p, ln_alpha_pm)
            ln_alpha_m = logsubexp(ln_alpha_m, ln_alpha_pm)

        ln_wkv_p = logaddexp(u + kt + ln_v_p, ln_alpha_p) - logaddexp(u + kt, ln_beta)
        ln_wkv_m = logaddexp(u + kt + ln_v_m, ln_alpha_m) - logaddexp(u + kt, ln_beta)

        wkv = torch.exp(ln_wkv_p) - torch.exp(ln_wkv_m)
        wkvs.append(wkv)

        ln_alpha_p = logaddexp(w + ln_alpha_p, kt + ln_v_p)
        ln_alpha_m = logaddexp(w + ln_alpha_m, kt + ln_v_m)
        ln_beta = logaddexp(w + ln_beta, kt)

        ln_alpha_ps.append(ln_alpha_p)
        ln_alpha_ms.append(ln_alpha_m)
        ln_betas.append(ln_beta)

    ln_alpha_p = torch.stack(ln_alpha_ps, dim=2)
    ln_alpha_m = torch.stack(ln_alpha_ms, dim=2)
    ln_beta = torch.stack(ln_betas, dim=2)

    return torch.cat(wkvs, 1), torch.cat((ln_alpha_p, ln_alpha_m, ln_beta), dim=1)
```

## Gradients

> This section implements a manual backward pass for the vanilla WKV computation [described above](#math), first derived from the equations above, then [implemented in PyTorch](#pytorch-implementation-3).

When we are actually implementing this in PyTorch, we will want to write an optimized kernel for performing the WKV computation. The downside is that it means we can't use autograd to compute the gradients for us, so we need to derive equations for the gradients.

Revisiting our original equation for the WKV computation:

$$\text{wkv}_i = \frac{ e^{u+k_i} v_i + \alpha_{i - 1} }{ e^{u+k_i} + \beta_{i - 1} }$$

The partial derivatives of the WKV computation with respect to $u$, $k$, and $v$ are as follows:

$$
\begin{aligned}
\frac{\partial \text{wkv}_i}{\partial u} = \frac{\partial \text{wkv}_i}{\partial k_i} & = \frac{ e^{u + k_i} v_i}{e^{u + k_i} + \beta_{i - 1}} - \frac{ e^{u + k_i} (e^{u + k_i} v_i + \alpha_{i - 1})}{(e^{u + k_i} + \beta_{i - 1})^2} \\
& = \frac{ e^{u + k_i} (\beta_{i - 1} v_i - \alpha_{i - 1})}{(\beta_{i - 1} + e^{u + k_i})^2} \\[1.5em]
\frac{\partial \text{wkv}_i}{\partial v_i} & = \frac{ e^{u + k_i}}{e^{u + k_i} + \beta_{i-1}} \\[1.5em]
\frac{\partial \text{wkv}_i}{\partial \alpha_{i-1}} & = \frac{1}{e^{u + k_i} + \beta_{i-1}} \\
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
\frac{\partial \alpha_i}{\partial \alpha_{i-1}} & = e^{w} \;
& \frac{\partial \alpha_i}{\partial v_i} & = e^{k_i} \\
\end{aligned}
$$

For $\beta_i$ we have:

$$
\begin{aligned}
\frac{\partial \beta_i}{\partial w} & = e^{w} \beta_{i - 1} \;
& \frac{\partial \beta_i}{\partial k_i} & = e^{k_i} \\[1.5em]
\frac{\partial \beta_i}{\partial \beta_{i-1}} & = e^{w} \;
& & \\
\end{aligned}
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
        alpha_prev, beta_prev = alpha[:, :, t], beta[:, :, t]
        euk = torch.exp(u + kt)
        ek = torch.exp(kt)

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

## Numerically Stable Gradients

> This section implements a manual backward pass for the numerically stable WKV computation [described above](#numerical-stability), first derived from the equations above, then [implemented in PyTorch](#pytorch-implementation-4).

Recall the numerically stable version of our WKV computation:

$$
\begin{aligned}
\alpha_i' & = e^{w + \epsilon_{i - 1} - \epsilon_i} \alpha_{i - 1}' + e^{k_i - \epsilon_i} v_i \\
\beta_i' & = e^{w + \epsilon_{i - 1} - \epsilon_i} \beta_{i - 1}' + e^{k_i - \epsilon_i} \\[1em]
\text{wkv}_i & = \frac{ e^{u+k_i-\tau_i} v_i + e^{\epsilon_{i - 1}-\tau_i}\alpha_{i - 1}' }{ e^{u+k_i-\tau_i} + e^{\epsilon_{i - 1}-\tau_i}\beta_{i - 1}' } \\[1em]
\epsilon_{i} & = \max(w + \epsilon_{i - 1}, k_i) \\
\tau_i & = \max(u + k_i, \epsilon_{i - 1}) \\
\end{aligned}
$$

The partial derivatives for the above computation are similar to the vanilla WKV computation, but adjusted for the $\epsilon$ terms:

$$
\begin{aligned}
\frac{\partial \text{wkv}_i'}{\partial u} = \frac{\partial \text{wkv}_i'}{\partial k_i} & = \frac{ e^{u + k_i - \tau_i} v_i}{e^{u + k_i - \tau_i} + e^{\epsilon_{i - 1} - \tau_i} \beta_{i - 1}'} - \frac{ e^{u + k_i - \tau_i} (e^{u + k_i - \tau_i} v_i + e^{\epsilon_{i - 1} - \tau_i} \alpha_{i - 1}')}{(e^{u + k_i - \tau_i} + e^{\epsilon_{i - 1} - \tau_i} \beta_{i - 1}')^2} \\
& = \frac{ e^{u + k_i - \tau_i} (e^{\epsilon_{i - 1} - \tau_i} \beta_{i - 1}' v_i - e^{\epsilon_{i - 1} - \tau_i} \alpha_{i - 1}')}{(e^{\epsilon_{i - 1} - \tau_i} \beta_{i - 1}' + e^{u + k_i - \tau_i})^2} \\[1.5em]
\frac{\partial \text{wkv}_i'}{\partial v_i} & = \frac{ e^{u + k_i - \tau_i}}{e^{u + k_i - \tau_i} + e^{\epsilon_{i - 1} - \tau_i} \beta_{i - 1}'} \\[1.5em]
\frac{\partial \text{wkv}_i'}{\partial \alpha_{i-1}'} & = \frac{e^{\epsilon_{i - 1} - \tau_i}}{e^{u + k_i - \tau_i} + e^{\epsilon_{i - 1} - \tau_i} \beta_{i - 1}'} \\[1.5em]
\frac{\partial \text{wkv}_i'}{\partial \beta_{i-1}'} & = -\frac{e^{\epsilon_{i-1} - \tau_i}(v_i e^{u + k_i - \tau_i} + e^{\epsilon_{i-1} - \tau_i}\alpha_{i-1}')}{(e^{u + k_i - \tau_i} + e^{\epsilon_{i - 1} - \tau_i} \beta_{i - 1}')^2} \\[1.5em]
\frac{\partial \text{wkv}_i'}{\partial \epsilon_{i - 1}} & = \frac{ e^{u + k_i + \epsilon_{i - 1}} (\alpha_{i - 1}' - v_i \beta_{i - 1}')}{(e^{\epsilon_{i - 1}} \beta_{i - 1}' + e^{u + k_i})^2}
= \frac{ e^{u + k_i + \epsilon_{i - 1} - 2 \tau_i} (\alpha_{i - 1}' - v_i \beta_{i - 1}')}{(e^{\epsilon_{i - 1} - \tau_i} \beta_{i - 1}' + e^{u + k_i - \tau_i})^2} \\
\end{aligned}
$$

For $\alpha_i'$ we have:

$$
\begin{aligned}
\frac{\partial \alpha_i'}{\partial w} = \frac{\partial \alpha_i'}{\partial \epsilon_{i-1}} & = e^{w + \epsilon_{i - 1} - \epsilon_i} \alpha_{i - 1}' \;
& \frac{\partial \alpha_i'}{\partial k_i} & = e^{k_i - \epsilon_i} v_i \\[1.5em]
\frac{\partial \alpha_i'}{\partial \epsilon_i} & = -\alpha_i' \;
& \frac{\partial \alpha_i'}{\partial \alpha_{i-1}'} & = e^{w + \epsilon_{i - 1} - \epsilon_i} \\[1.5em]
\frac{\partial \alpha_i'}{\partial v_i} & = e^{k_i - \epsilon_i} & & \\
\end{aligned}
$$

For $\beta_i'$ we have:

$$
\begin{aligned}
\frac{\partial \beta_i'}{\partial w} = \frac{\partial \beta_i'}{\partial \epsilon_{i-1}} & = e^{w + \epsilon_{i - 1} - \epsilon_i} \beta_{i - 1}' \;
& \frac{\partial \beta_i'}{\partial k_i} & = e^{k_i - \epsilon_i} \\[1.5em]
\frac{\partial \beta_i'}{\partial \epsilon_i} & = -\beta_i' \;
& \frac{\partial \beta_i'}{\partial \beta_{i-1}'} & = e^{w + \epsilon_{i - 1} - \epsilon_i} \\
\end{aligned}
$$

For $\epsilon_i$ we have:

$$
\begin{aligned}
\frac{\partial \epsilon_i}{\partial w} = \frac{\partial \epsilon_i}{\partial \epsilon_{i - 1}} & = \begin{cases} 1 & \text{if } w + \epsilon_{i - 1} > k_i \\ 0 & \text{otherwise} \end{cases} \\
\frac{\partial \epsilon_i}{\partial k_i} & = \begin{cases} 1 & \text{if } w + \epsilon_{i - 1} < k_i \\ 0 & \text{otherwise} \end{cases} \\
\end{aligned}
$$

### PyTorch Implementation

The PyTorch implementation for the numerically stable gradients will be similar to the vanilla gradients, but with the addition of the $\epsilon$ terms.

```python
def wkv_with_eps_backward(
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
    assert state.shape == (bsz, 3, tsz + 1, chans)
    assert grad_wkv.shape == (bsz, tsz, chans)
    assert grad_state.shape == (bsz, 3, 1, chans)

    alpha, beta, eps = state.chunk(3, dim=1)  # (B, 1, T + 1, D), (B, 1, T + 1, D), (B, 1, T + 1, D)
    grad_alpha, grad_beta, grad_eps = grad_state[:, :, 0].chunk(3, dim=1)  # (B, 1, D), (B, 1, D), (B, 1, D)
    grad_eps = grad_eps.clone()

    grad_w = torch.zeros_like(w)
    grad_u = torch.zeros_like(u)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    for t in reversed(range(tsz)):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        alpha_prev, beta_prev, eps_prev = alpha[:, :, t], beta[:, :, t], eps[:, :, t]
        alpha_curr, beta_curr, eps_curr = alpha[:, :, t + 1], beta[:, :, t + 1], eps[:, :, t + 1]
        ukt = u + kt
        tau = torch.maximum(ukt, eps_prev)
        e1 = torch.exp(eps_prev - tau)
        e2 = torch.exp(ukt - tau)

        euke = torch.exp(ukt + eps_prev - 2 * tau)

        denom = e1 * beta_prev + e2
        denom_sq = denom**2

        grad_wkvt = grad_wkv[:, t : t + 1]

        # Backpropagates wkv gradients.
        grad_uk = grad_wkvt * e2 * (e1 * beta_prev * vt - e1 * alpha_prev) / denom_sq
        grad_u += grad_uk.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_uk
        grad_v[:, t : t + 1] += grad_wkvt * e2 / denom

        grad_alpha_wkv = grad_wkvt * e1 / denom
        grad_beta_wkv = -grad_wkvt * e1 * (e2 * vt + e1 * alpha_prev) / denom_sq
        grad_eps_wkv = grad_wkvt * euke * (alpha_prev - vt * beta_prev) / (e1 * beta_prev + e2) ** 2

        e1 = torch.exp(w + eps_prev - eps_curr)
        e2 = torch.exp(kt - eps_curr)

        # Backpropagates alpha gradients.
        grad_alpha_we = grad_alpha * e1 * alpha_prev
        grad_w += grad_alpha_we.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_alpha * e2 * vt
        grad_v[:, t : t + 1] += grad_alpha * e2
        grad_eps += grad_alpha * -alpha_curr

        # Backpropagates beta gradients.
        grad_beta_we = grad_beta * e1 * beta_prev
        grad_w += grad_beta_we.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_beta * e2
        grad_eps += grad_beta * -beta_curr

        # Backpropagates epsilon gradients.
        eps_grad_mask = w + eps_prev > kt
        grad_eps_we = torch.where(eps_grad_mask, grad_eps, torch.zeros_like(grad_eps))
        grad_w += grad_eps_we.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += torch.where(eps_grad_mask, torch.zeros_like(grad_eps), grad_eps)

        # Computes gradients for alpha, beta and epsilon.
        grad_alpha = grad_alpha * e1 + grad_alpha_wkv
        grad_beta = grad_beta * e1 + grad_beta_wkv
        grad_eps = grad_alpha_we + grad_beta_we + grad_eps_we + grad_eps_wkv

    return grad_w, grad_u, grad_k, grad_v, torch.stack((grad_alpha, grad_beta, grad_eps), dim=1)
```

## Log-Space Gradients

In our log-space implementation of the WKV computation, we have:

$$\text{wkv}_i = e^{\log \text{wkv}_i^+} - e^{\log \text{wkv}_i^-}$$

This gives us the following partial derivatives:

$$
\begin{aligned}
\frac{\partial \text{wkv}_i}{\partial \log \text{wkv}_i^+} & = \text{wkv}_i^+ \\[1.5em]
\frac{\partial \text{wkv}_i}{\partial \log \text{wkv}_i^-} & = -\text{wkv}_i^- \\
\end{aligned}
$$

Next, we need to compute the gradients of $\log \text{wkv}_i^+$ and $\log \text{wkv}_i^-$ with respect to each of our inputs. We have:

$$
\begin{aligned}
\log \text{wkv}_i^+ & = LSE(u + k_i + \log v_i^+, \log \alpha_{i - 1}^+) - LSE(u + k_i, \log \beta_{i - 1}) \\[1em]
\log \text{wkv}_i^- & = LSE(u + k_i + \log v_i^-, \log \alpha_{i - 1}^-) - LSE(u + k_i, \log \beta_{i - 1}) \\
\end{aligned}
$$

These two equations are identical, so in the equations below we omit the sign and simply use $\log \text{wkv}_i$.

The gradients of the log-sum-exp function are given by:

$$
\begin{aligned}
\frac{\partial LSE(a, b)}{\partial a} & = \frac{e^a}{e^a + e^b} & = \frac{1}{1 + e^{b - a}} \\[1em]
\frac{\partial LSE(a, b)}{\partial b} & = \frac{e^b}{e^a + e^b} & = \frac{1}{1 + e^{a - b}} \\
\end{aligned}
$$

We can use this to find the partial derivatives. Note that since we are using log-space state variables, we need to find the partial derivatives with respect to $\log \alpha_{i-1}$ and $\log \beta_{i-1}$ rather than $\alpha_{i-1}$ and $\beta_{i-1}$. We avoid simplifying the expression because it more closely matches the implementation in code.

$$
\begin{aligned}
\frac{\partial \log \text{wkv}_i}{\partial u_i} = \frac{\partial \log \text{wkv}_i}{\partial k_i} & = \frac{1}{1 + e^{\log \alpha_{i - 1} - (u + k_i + \log v_i)}} - \frac{1}{1 + e^{\log \beta_{i - 1} - (u + k_i)}} \\[1em]
\frac{\partial \log \text{wkv}_i}{\partial v_i} & = \frac{1}{v_i (1 + e^{\log \alpha_{i - 1} - (u + k_i + \log v_i)})} \\[1em]
\frac{\partial \log \text{wkv}_i}{\partial \log \alpha_{i - 1}} & = \frac{1}{1 + e^{(u + k_i + \log v_i) - \log \alpha_{i - 1}}} \\[1em]
\frac{\partial \log \text{wkv}_i}{\partial \log \beta_{i - 1}} & = -\frac{1}{1 + e^{(u + k_i) - \log \beta_{i - 1}}} \\[1em]
\end{aligned}
$$

Additionally, we need to find the partial derivatives of $\log \alpha_{i}$ and $\log \beta_{i}$. Recall the log-space update rule:

$$
\begin{aligned}
\log \alpha_i & = LSE(w + \log \alpha_{i-1}, k_i + \log v_i) \\
\log \beta_i & = LSE(w + \log \beta_{i - 1}, k_i) \\
\end{aligned}
$$

The partial derivatives of $\log \alpha_i$ are:

$$
\begin{aligned}
\frac{\partial \log \alpha_i}{\partial w} = \frac{\partial \log \alpha_i}{\partial \log \alpha_{i - 1}} & = \frac{1}{1 + e^{(k_i + \log v_i) - (w + \log \alpha_{i - 1})}} \\[1em]
\frac{\partial \log \alpha_i}{\partial k_i} & = \frac{1}{1 + e^{(\log \alpha_{i - 1} + w) - (k_i + \log v_i)}} \\[1em]
\frac{\partial \log \alpha_i}{\partial v_i} & = \frac{1}{v_i (1 + e^{(\log \alpha_{i - 1} + w) - (k_i + \log v_i)})} \\
\end{aligned}
$$

The partial derivatives of $\log \beta_{i}$ are:

$$
\begin{aligned}
\frac{\partial \log \beta_i}{\partial w} = \frac{\partial \log \beta_i}{\partial \log \beta_{i - 1}} & = \frac{1}{1 + e^{k_i - (w + \log \beta_{i - 1})}} \\[1em]
\frac{\partial \log \beta_i}{\partial k_i} & = \frac{1}{1 + e^{(w + \log \beta_{i - 1}) - k_i}} \\
\end{aligned}
$$

Lastly, a small point of note regarding the partial derivatives of $v_i$:

$$
\begin{aligned}
\frac{\partial v_i^+}{\partial v_i} & = \begin{cases} 1 & \text{if } v_i > 0 \\ 0 & \text{otherwise} \end{cases} \\
\frac{\partial v_i^-}{\partial v_i} & = \begin{cases} -1 & \text{if } v_i < 0 \\ 0 & \text{otherwise} \end{cases} \\
\end{aligned}
$$

### PyTorch Implementation

The PyTorch implementation follows from the equation above, although there is some trickiness involved in dealing with the positive and negative sides.

```python
def wkv_log_space_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
    eps: float = EPS,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 3, tsz + 1, chans)
    assert grad_wkv.shape == (bsz, tsz, chans)
    assert grad_state.shape == (bsz, 3, 1, chans)

    grad_ln_alpha_p, grad_ln_alpha_m, grad_ln_beta = grad_state[:, :, 0].chunk(3, dim=1)

    grad_w = torch.zeros_like(w)
    grad_u = torch.zeros_like(u)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    def logaddexp(a: Tensor, b: Tensor) -> Tensor:
        max_av = torch.maximum(a, b)
        return max_av + torch.log(torch.exp(a - max_av) + torch.exp(b - max_av))

    for t in reversed(range(tsz)):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        vt_p, vt_m = torch.clamp_min(vt, 0) + eps, torch.clamp_min(-vt, 0) + eps
        ln_v_p, ln_v_m = torch.log(vt_p), torch.log(vt_m)

        ln_alpha_p_prev, ln_alpha_m_prev, ln_beta_prev = state[:, :, t].chunk(3, dim=1)

        uk = u + kt
        ukv_p, ukv_m = uk + ln_v_p, uk + ln_v_m

        ukb = logaddexp(uk, ln_beta_prev)
        wkv_p = torch.exp(logaddexp(ukv_p, ln_alpha_p_prev) - ukb)
        wkv_m = torch.exp(logaddexp(ukv_m, ln_alpha_m_prev) - ukb)

        grad_wkvt = grad_wkv[:, t : t + 1]
        grad_ln_wkv_p, grad_ln_wkv_m = grad_wkvt * wkv_p, grad_wkvt * -wkv_m

        # Backpropagates wkv gradients.
        e_num_p = torch.exp(ln_alpha_p_prev - ukv_p)
        e_num_m = torch.exp(ln_alpha_m_prev - ukv_m)
        e_den = torch.exp(ln_beta_prev - uk)
        grad_wkv_den_p, grad_wkv_den_m = grad_ln_wkv_p / (1 + e_den), grad_ln_wkv_m / (1 + e_den)
        grad_kv_p, grad_kv_m = grad_ln_wkv_p / (1 + e_num_p), grad_ln_wkv_m / (1 + e_num_m)
        grad_uk = grad_kv_p + grad_kv_m - grad_wkv_den_p - grad_wkv_den_m
        grad_u += grad_uk.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_uk
        grad_v[:, t : t + 1] += torch.where(vt > 0, grad_kv_p / vt_p, grad_kv_m / -vt_m)

        grad_ln_alpha_wkv_p = grad_ln_wkv_p / (1 + (1 / e_num_p))
        grad_ln_alpha_wkv_m = grad_ln_wkv_m / (1 + (1 / e_num_m))
        grad_ln_beta_wkv = -grad_ln_wkv_p / (1 + (1 / e_den)) - grad_ln_wkv_m / (1 + (1 / e_den))

        # Backpropagates alpha gradients.
        e_alpha_p = torch.exp(kt + ln_v_p - (w + ln_alpha_p_prev))
        e_alpha_m = torch.exp(kt + ln_v_m - (w + ln_alpha_m_prev))
        grad_wa_p = grad_ln_alpha_p / (1 + e_alpha_p)
        grad_wa_m = grad_ln_alpha_m / (1 + e_alpha_m)
        grad_w += (grad_wa_p + grad_wa_m).flatten(0, -2).sum(0)
        grad_kv_p, grad_kv_m = grad_ln_alpha_p / (1 + (1 / e_alpha_p)), grad_ln_alpha_m / (1 + (1 / e_alpha_m))
        grad_k[:, t : t + 1] += grad_kv_p + grad_kv_m
        grad_v[:, t : t + 1] += torch.where(vt > 0, grad_kv_p / vt_p, -grad_kv_m / vt_m)

        # Backpropagates beta gradients.
        e_beta = torch.exp(kt - (w + ln_beta_prev))
        grad_wb = grad_ln_beta / (1 + e_beta)
        grad_w += grad_wb.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_ln_beta / (1 + (1 / e_beta))

        # Compute gradients for log alpha and log beta.
        grad_ln_alpha_p = grad_wa_p + grad_ln_alpha_wkv_p
        grad_ln_alpha_m = grad_wa_m + grad_ln_alpha_wkv_m
        grad_ln_beta = grad_wb + grad_ln_beta_wkv

    return grad_w, grad_u, grad_k, grad_v, torch.stack((grad_ln_alpha_p, grad_ln_alpha_m, grad_ln_beta), dim=1)
```

{% endkatexmm %}

[rwkv-model]: https://github.com/BlinkDL/RWKV-LM
[blog-post]: https://johanwind.github.io/2023/03/23/rwkv_details.html
[minimal-inference]: https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_in_150_lines.py
[log-sum-exp]: https://en.wikipedia.org/wiki/LogSumExp
[rwkv-repo]: https://github.com/codekansas/rwkv
