---
layout: post
title: Recent Speech Papers
tags: [ml, refs, speech]
excerpt: >
  A collection of summaries of recent papers about speech.
---

## Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport

- [Arxiv](https://arxiv.org/pdf/2302.00482.pdf)
- [Github](https://github.com/atong01/conditional-flow-matching)
- [PapersWithCode](https://paperswithcode.com/paper/conditional-flow-matching-simulation-free)

## Flow Matching for Generative Modeling

- [Arxiv](https://arxiv.org/pdf/2210.02747.pdf)
- [PapersWithCode](https://paperswithcode.com/paper/flow-matching-for-generative-modeling)

## High Fidelity Neural Audio Compression

- [Arxiv](https://arxiv.org/pdf/2210.13438.pdf)
- [Github](https://github.com/facebookresearch/encodec)

Encodec paper

## Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale

- [Release post](https://research.facebook.com/publications/voicebox-text-guided-multilingual-universal-speech-generation-at-scale/)
- [Another release post](https://ai.facebook.com/blog/voicebox-generative-ai-model-speech/)

- Some datasets that other papers have tried:
  - Librispeech
  - LibriTTS
  - CommonVoice
- Synthetic data from VoiceBox can be used for training good-quality ASR systems

## ERNIE-Music: Text-to-Waveform Music Generation with Diffusion Models

- [Arxiv](https://arxiv.org/pdf/2302.04456.pdf)

## Hybrid Spectrogram and Waveform Source Separation

- [Arxiv](https://arxiv.org/pdf/2111.03600.pdf)

Original demucs paper

## Hybrid Transformers for Music Source Separation

- [Arxiv](https://arxiv.org/pdf/2211.08553.pdf)
- [Github](https://github.com/facebookresearch/demucs)

Demucs improvement paper

## SoundStorm: Efficient Parallel Audio Generation

- [Arxiv](https://arxiv.org/pdf/2305.09636.pdf)
- [Write-up](https://google-research.github.io/seanet/soundstorm/examples/)
- [lucidrains Implementation](https://github.com/lucidrains/soundstorm-pytorch)

## NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers

- [Arxiv](https://arxiv.org/pdf/2304.09116.pdf)
- [Write-up](https://speechresearch.github.io/naturalspeech2/)
- [lucidrains Implementation](https://github.com/lucidrains/naturalspeech2-pytorch)

## AudioLM: a Language Modeling Approach to Audio Generation

- [Arxiv](https://arxiv.org/pdf/2209.03143.pdf)
- [Write-up](https://google-research.github.io/seanet/audiolm/examples/)
- [lucidrains Implementation](https://github.com/lucidrains/audiolm-pytorch)

## MusicLM: Generating Music From Text

- [Arxiv](https://arxiv.org/pdf/2301.11325.pdf)
- [Write-up](https://google-research.github.io/seanet/musiclm/examples/)
- [lucidrains Implementation](https://github.com/lucidrains/musiclm-pytorch)

## CycleGAN Voice Converter

- [Write-up](https://leimao.github.io/project/Voice-Converter-CycleGAN/)
- [Github](https://github.com/leimao/Voice-Converter-CycleGAN)

This paper is pretty old (about 5 years old), but the results are pretty relevant to stuff I'm working on right now so I figured I should write up a summary of what it's doing.

### TL;DR

- CycleGAN on Mel-cepstral coefficients (MCEPs)
