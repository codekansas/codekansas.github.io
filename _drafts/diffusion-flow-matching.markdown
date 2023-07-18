---
layout: post
title: Diffusion and Flow Matching for Speech
tags: [ml, speech]
excerpt: >
  An accessible introduction to diffusion and flow matching models.
---

{% katexmm %}

This post is a summary of the various generative speech modeling papers I've seen come out recently. I'm primarily writing this for my own understanding, but hopefully it's useful to others as well, since this is a fast-paced area of research and it can be hard to dive deeply into each new paper that comes out.

This post uses what I'll call "math for computer scientists", meaning there will likely be a lot of abuse of notation and other hand-waving with the goal of conveying the underlying idea more clearly. If there is a mistake (and it looks unintentional) then let me know!

## All Papers

### Images

- [Diffusion][diffusion-paper] from _Denoising Diffusion Probabilistic Models_
- [Stable Diffusion][latent-diffusion-paper] from _High-Resolution Image Synthesis with Latent Diffusion Models_
- [Visual ChatGPT][visual-chatgpt-paper] from _Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models_
- [Flow Matching][flow-matching-paper] from _Flow Matching for Generative Modeling_
- [Autoregressive Diffusion][autoregressive-diffusion-paper] from _Autoregressive Diffusion Models_

### Speech

- [Voicebox][voicebox-paper] from _Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale_
- [MMS][mms-paper] from _Scaling Speech Technology to 1,000+ Languages_

### Other Good References

- [What are diffusion models?][diffusion-lillog] from _Lil' Log_
- [Diffusion Models from Scratch][diffusion-xinduan] from _Tony Duan_
- [Collection of Speech Synthesis Papers][speech-synthesis-papers]

These papers overlap and cite each other in various ways.

## Diffusion

Diffusion models can be summarized very shortly as:

1. Start with a random image.
2. Iteratively add noise to the image.
3. Train a model to predict some part of the noise that was added.
4. Iteratively remove noise from the image by predicting and subtracting it.

Consider Figure 2 from the original paper:

![Figure 2 from the original flow matching paper.](/images/diffusion-flow-matching/ddpm-fig-2.webp)

In the above diagram:

- $x_T$ is some noise, typically Gaussian noise
- $x_0$ is the original image
- $p_{\theta}(x_{t-1}|x_t)$ is called the _reverse process_, a distribution over the sligtly less noisy image given the current image
- $q(x_t|x_{t-1})$ is called the _forward process_, a Markov chain that adds Gaussian noise to the data

### How do we convert a regular image to Gaussian noise?

Let's assume we have a noisy image $x_{t-1}$ and we want to make it slightly noisier (in other words, take a step along the _forward process_, $q(x_t|x_{t-1})$). We sample the slightly noisier image $x_t$ from the following distribution:

$$x_t \sim \mathcal{N}(\sqrt{1 - \beta_t} x_{t - 1}, \beta_t \textbf{I})$$

This can be read as, "sample $x_t$ from a Gaussian distribution with mean $\sqrt{1 - \beta_t} x_{t - 1}$ and variance $\beta_t \textbf{I}$." The matrix $\textbf{I}$ is just the identity matrix, so $\beta_t$ is the variance of the Gaussian noise being added.

Recall that the sum of two Gaussian distributions is also a Gaussian distribution:

$$
\begin{aligned}
X & \sim \mathcal{N}(\mu_Y, \sigma_X^2) \\
Y & \sim \mathcal{N}(\mu_Y, \sigma_Y^2) \\
X + Y & \sim \mathcal{N}(\mu_X + \mu_Y, \sigma_X^2 + \sigma_Y^2)
\end{aligned}
$$

It's also worth noting that multiplying a zero-mean Gaussian by some factor $\alpha$ is equivalent to multiplying the variance by $\alpha^2$:

$$\alpha \mathcal{N}(\textbf{0}, \textbf{I}) = \mathcal{N}(\textbf{0}, \alpha^2 \textbf{I})$$

So we can rewrite the distribution from earlier as:

$$
\begin{aligned}
\mathcal{N}(\sqrt{1 - \beta_t} x_{t - 1}, \beta_t \textbf{I}) & = \mathcal{N}(\sqrt{1 - \beta_t} x_{t - 1}, \textbf{0}) + \mathcal{N}(\textbf{0}, \beta_t \textbf{I}) \\
& = \sqrt{1 - \beta_t} x_{t - 1} + \sqrt{\beta_t} \mathcal{N}(\textbf{0}, \textbf{I})
\end{aligned}
$$

This is our _forward process_ $q(x_t|x_{t-1})$:

$$q(x_t|x_{t-1}) = \sqrt{1 - \beta_t} x_{t - 1} + \sqrt{\beta_t} \mathcal{N}(\textbf{0}, \textbf{I})$$

So, to recap, in order to convert a regular image to Gaussian noise, we repeatedly apply the $q(x_t|x_{t-1})$ rule to add noise to the image, and $T \to \infty$ will result in a random distribution of noise.

The variances for each step of the $q$ update is given by some schedule $\beta_1, \beta_2, \dots, \beta_T$. The schedule is typically linearly increasing from 0 to 1, so that on the final step when $\beta_T = 1$ we will sample a completely noisy image from the distribution $\mathcal{N}(\textbf{0}, \textbf{I})$.

### How do you sample $x_t$ in closed form (i.e., without sampling $x_{t - 1}, ..., x_{1}$)?

Rather than having to take our original image and run 1 to $T$ steps to get a noisy image, we can use the _reparametrization trick_.

If we start with our original image $x_0$, we can write the first slightly noisy image $x_1$ as:

$$
\begin{aligned}
x_1 & = \sqrt{1 - \beta_1} x_0 + \mathcal{N}(\textbf{0}, \beta_1 \textbf{I}) \\
& = \sqrt{1 - \beta_1} x_0 + \sqrt{\beta_1} \mathcal{N}(\textbf{0}, \textbf{I}) \\
\end{aligned}
$$

We can rewrite this using $\alpha_t = 1 - \beta_t$ as:

$$x_1 = \sqrt{\alpha_1} x_0 + \mathcal{N}(\textbf{0}, (1 - \alpha_1) \textbf{I})$$

We can then write $x_2$ as:

$$
\begin{aligned}
x_2 & = \sqrt{\alpha_2} x_1 + \mathcal{N}(\textbf{0}, (1 - \alpha_2) \textbf{I}) \\
& = \sqrt{\alpha_2} (\sqrt{\alpha_1} x_0 + \mathcal{N}(\textbf{0}, (1 - \alpha_1) \textbf{I})) + \mathcal{N}(\textbf{0}, (1 - \alpha_2) \textbf{I}) \\
& = \sqrt{\alpha_1 \alpha_2} x_0 + \mathcal{N}(\textbf{0}, \alpha_2 (1 - \alpha_1) \textbf{I}) + \mathcal{N}(\textbf{0}, (1 - \alpha_2) \textbf{I}) \\
& = \sqrt{\alpha_1 \alpha_2} x_0 + \mathcal{N}(\textbf{0}, (\alpha_2 (1 - \alpha_1) + (1 - \alpha_2)) \textbf{I}) \\
& = \sqrt{\alpha_1 \alpha_2} x_0 + \mathcal{N}(\textbf{0}, (1 - \alpha_1 \alpha_2) \textbf{I}) \\
\end{aligned}
$$

This can be extended recursively[^1], so we can write $x_t$ in **closed form** as:

$$x_t = \sqrt{\alpha_1 \alpha_2 \dots \alpha_t} x_0 + \mathcal{N}(\textbf{0}, (1 - \alpha_1 \alpha_2 \dots \alpha_t) \textbf{I})$$

It's common to express the product as a new variable:

$$\bar{\alpha}_t = \alpha_1 \alpha_2 \dots \alpha_t = \prod_{i=1}^{t} \alpha_i$$

Also, the usual notation is to write $\epsilon_t \sim \mathcal{N}(\textbf{0}, \textbf{I})$, giving the final equation for sampling $x_t$ as:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t$$

Sampling $\epsilon_t$ from $\mathcal{N}(\textbf{0}, \textbf{I})$ and using it to derive $x_t$ is called _Monte Carlo sampling_. Alternatively, we can use our $q$ notation from earlier to specify the closed-form distribution that the sample is drawn from:

$$
\begin{aligned}
q(x_t | x_0) & = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \mathcal{N}(\textbf{0}, \textbf{I}) \\
& = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \textbf{I}) \\
\end{aligned}
$$

### How is the model trained?

![The diffusion model training and sampling algorithms](/images/diffusion-flow-matching/ddpm-algs.webp)

The main goal of the learning process is to maximize the likelihood of the data after repeatedly applying the reverse process. First, we sample some noise $\epsilon_t \sim \mathcal{N}(\textbf{0}, \textbf{I})$ and then we apply the forward process $q(x_t | x_0)$ to get $x_t$:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t$$

The diffusion model training process involves training a model which takes $x_t$ and $t$ as input and predicts $n_t$:

$$\hat{\epsilon}_t = \epsilon_{\theta}(x_t, t)$$

We can train the model to minimize the mean squared error between $\epsilon_t$ and $\hat{\epsilon}_t$:

$$\mathcal{L} = ||\epsilon_t - \hat{\epsilon}_t||^2$$

So, the model is predicting the _noise_ between the _noisy_ image and the _original_ image.

### How do you sample from the model?

Now that we've ironed out the math for the _forward_ process, we need to flip it around to get the _reverse_ process. In other words, given that we have $q(x_t | x_{t-1})$, we need to derive $q(x_{t-1} | x_t)$ [^2]. The first step is to apply the chain rule:

$$
\begin{aligned}
q(x_{t-1} | x_t) & = \frac{q(x_t | x_{t-1}) q(x_{t-1})}{q(x_t)} \\
& \propto q(x_t | x_{t-1}) q(x_{t-1}) \\
\end{aligned}
$$

We drop the denominator because $x_t$ is constant when we are sampling.

Recall the probability density function for a normal distribution:

$$\mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp(-\frac{(x - \mu)^2}{2 \sigma^2})$$

We can use this to rewrite $q(x_t | x_{t-1})$ as a function of $x_{t-1}$:

$$
\begin{aligned}
q(x_t | x_{t-1}) & = \mathcal{N}(x_t | \sqrt{\bar{\alpha}_t} x_{t-1}, (1 - \bar{\alpha}_t) \textbf{I}) \\
& = \frac{1}{\sqrt{2 \pi (1 - \bar{\alpha}_t)}} \exp(-\frac{(x_t - \sqrt{\bar{\alpha}_t} x_{t-1})^2}{2 (1 - \bar{\alpha}_t)}) \\
\end{aligned}
$$

Similarly for $q(x_{t-1})$:

$$
\begin{aligned}
q(x_{t-1}) & = \mathcal{N}(x_{t-1} | \sqrt{\bar{\alpha}_{t-1}} x_0, (1 - \bar{\alpha}_{t-1}) \textbf{I}) \\
& = \frac{1}{\sqrt{2 \pi (1 - \bar{\alpha}_{t-1})}} \exp(-\frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0)^2}{2 (1 - \bar{\alpha}_{t-1})}) \\
\end{aligned}
$$

Anyway, somehow if you do some crazy math you can eventually arrive at the equations for the forward process, which are:

$$x_{t - 1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{\theta}(x_t, t)) + \sigma_t z$$

where $z \sim \mathcal{N}(\textbf{0}, \textbf{I})$ is new noise to add at each step and $\epsilon_{\theta}(x_t, t)$ is the output of the model.

[^1]: Proof by "trust me, bro"
[^2]: Alternatively denoted $p(x_{t-1} | x_t)$ so that you can just use the $q$ function everywhere

{% endkatexmm %}

[autoregressive-diffusion-paper]: https://arxiv.org/pdf/2110.02037v2.pdf
[diffusion-lillog]: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
[diffusion-paper]: https://arxiv.org/abs/2006.11239
[diffusion-xinduan]: https://www.tonyduan.com/diffusion.html
[flow-matching-paper]: https://arxiv.org/abs/2210.02747
[latent-diffusion-paper]: https://arxiv.org/pdf/2112.10752.pdf
[mms-paper]: https://arxiv.org/pdf/2305.13516.pdf
[speech-synthesis-papers]: https://github.com/wenet-e2e/speech-synthesis-paper
[visual-chatgpt-paper]: https://arxiv.org/pdf/2303.04671.pdf
[voicebox-paper]: https://research.facebook.com/publications/voicebox-text-guided-multilingual-universal-speech-generation-at-scale/
[xkcd-decorative]: https://xkcd.com/2566/
