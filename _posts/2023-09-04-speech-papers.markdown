---
layout: post
title: Recent Speech Papers
tags: [ml, refs, speech]
excerpt: >
  A collection of summaries of recent papers about speech.
---

## Flow Matching

Flow matching is a simpler and arguably more effective version of diffusion modeling which has recently been applied to speech. The standard approach is to train a flow matching model on the Mel spectrograms and then reconstruct it using a vocoder like HiFi-GAN. I wrote up a summary of flow matching [here]({% post_url 2023-07-19-diffusion-flow-matching %}).

### Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport

- [Arxiv](https://arxiv.org/pdf/2302.00482.pdf)
- [Github](https://github.com/atong01/conditional-flow-matching)
- [PapersWithCode](https://paperswithcode.com/paper/conditional-flow-matching-simulation-free)

This was the original paper which introduced the idea flow matching.

### Flow Matching for Generative Modeling

- [Arxiv](https://arxiv.org/pdf/2210.02747.pdf)
- [PapersWithCode](https://paperswithcode.com/paper/flow-matching-for-generative-modeling)

This paper gets flow matching to work for real data.

### Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale

- [Release post](https://research.facebook.com/publications/voicebox-text-guided-multilingual-universal-speech-generation-at-scale/)
- [Another release post](https://ai.facebook.com/blog/voicebox-generative-ai-model-speech/)

This paper extends flow matching to speech with very high-quality results, solving a bunch of text-to-speech and speech-to-text tasks with a single infilling model. One interesting highlight from this paper is that the speech quality is so good that you can train an ASR model on the synthetic speech which performs better than ASR models trained on _real_ speech.

## Vector Quantization

Vector quantization is an increasingly popular technique for generative modeling. You first learn a near-lossless autoencoder which goes through some set of tokens. You can then build models on this lower-dimensional token space using language modeling approaches (since they're just fixed tokens). This has worked pretty well for image modeling so it makes sense to use it for modeling audio as well (or increasingly just modeling all modalities together).

A good library for doing vector quantization is the one from `lucidrains` [here](https://github.com/lucidrains/vector-quantize-pytorch).

### High Fidelity Neural Audio Compression

- [Arxiv](https://arxiv.org/pdf/2210.13438.pdf)
- [Github](https://github.com/facebookresearch/encodec)

Encodec paper. The idea here is to use residual vector quantization to learn a set of discrete tokens which can then be used to reconstruct the full audio. There's a couple ways to do vector quantization, but the method used in this paper (codebook learning with some clever techniques for learning good codebooks) is pretty effective compared to other approaches such as the Gumbel Softmax.

### AudioLM: a Language Modeling Approach to Audio Generation

- [Arxiv](https://arxiv.org/pdf/2209.03143.pdf)
- [Write-up](https://google-research.github.io/seanet/audiolm/examples/)
- [lucidrains Implementation](https://github.com/lucidrains/audiolm-pytorch)

### MusicLM: Generating Music From Text

- [Arxiv](https://arxiv.org/pdf/2301.11325.pdf)
- [Write-up](https://google-research.github.io/seanet/musiclm/examples/)
- [lucidrains Implementation](https://github.com/lucidrains/musiclm-pytorch)

## Source Separation

### Hybrid Spectrogram and Waveform Source Separation

- [Arxiv](https://arxiv.org/pdf/2111.03600.pdf)

Demucs paper. This paper introduces a model which is used for doing source separation for real-time background noise reduction, with pre-trained weights. You can run the model yourself pretty easily, it runs in real-time on a normal laptop.

### Hybrid Transformers for Music Source Separation

- [Arxiv](https://arxiv.org/pdf/2211.08553.pdf)
- [Github](https://github.com/facebookresearch/demucs)

Demucs improvement paper, which extends the architecture for separate music into the singer plus different instruments.

## Multi-Modal

### LLaSM: Large Language and Speech Model

- [HuggingFace](https://huggingface.co/papers/2308.15930)
- [Github](https://github.com/LinkSoul-AI/LLaSM)
- [Project Page](https://huggingface.co/spaces/LinkSoul/LLaSM)
- [Dataset](https://huggingface.co/datasets/LinkSoul/LLaSM-Audio-Instructions)

Takes a pre-trained Whisper audio encoder and a pre-trained LLaMa model (with both Chinese and English) and fine-tunes end-to-end on a dataset generated using GPT-4, ShareGPT, and WizardLM. Uses text-to-speech to generate the speech component.

![LLaSM Model Architecture](/images/speech-papers/llasm.webp)

### Visual Instruction Tuning

- [HuggingFace](https://huggingface.co/papers/2304.08485)
- [Project Page](https://llava-vl.github.io/)
- [Github](https://github.com/haotian-liu/LLaVA)
- [Weights](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0)
- [Demo](https://llava.hliu.cc/)

LLaVa paper. Takes a pre-trained CLIP visual encoder and a pre-trained LLaMa model, glues them together, and fine-tunes end-to-end on a dataset generated using GPT-4 (specifically, by providing the text descriptions of the image, from datasets like Coco, and asking GPT-4 to generate instruction-following tasks for what it knows about the images). They compared GPT-4 with ChatGPT and found that GPT-4 was better for what they were doing.

Here's the model architecture from Figure 1 of the paper.

![LLaVa Model Architecture](/images/speech-papers/llava.webp)

## CycleGAN

### CycleGAN Voice Converter

- [Write-up](https://leimao.github.io/project/Voice-Converter-CycleGAN/)
- [Github](https://github.com/leimao/Voice-Converter-CycleGAN)

This is an older paper but I found the demo pretty compelling so I figured I'd include it. The basic idea is to do CycleGAN on Mel-cepstral coefficients (MCEPs). I implemented this myself but it seems to only work well if you have two speakers with paired data, not with many-speakers to many-speakers. Scaling up GANs is hard.

## Datasets

These aren't papers, just datasets I've come across that seem like they could be useful for something in the future.

- [Spotify Podcasts Dataset](https://www.kaggle.com/datasets/tamle507/spotify-top-100-usa-podcasts-with-eps)
  - [Another link to the same dataset](https://www.kaggle.com/code/tamle507/dataset-spotify-top-100-usa-podcasts-with-eps)

## Backlog

Here's my paper backlog.

### SoundStorm: Efficient Parallel Audio Generation

- [Arxiv](https://arxiv.org/pdf/2305.09636.pdf)
- [Write-up](https://google-research.github.io/seanet/soundstorm/examples/)
- [lucidrains Implementation](https://github.com/lucidrains/soundstorm-pytorch)

### NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers

- [Arxiv](https://arxiv.org/pdf/2304.09116.pdf)
- [Write-up](https://speechresearch.github.io/naturalspeech2/)
- [lucidrains Implementation](https://github.com/lucidrains/naturalspeech2-pytorch)

### ERNIE-Music: Text-to-Waveform Music Generation with Diffusion Models

- [Arxiv](https://arxiv.org/pdf/2302.04456.pdf)
