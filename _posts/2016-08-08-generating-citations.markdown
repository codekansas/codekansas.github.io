---
layout: post
title: "Using TF-IDF to Annotate to a Trump Speech"
date: 2016-08-08 12:00:00
categories: machine-learning
excerpt: >
  Explaining how to use TF-IDF scores for document similarity and applying those to choose documents to cite for particular sentences.
---

* TOC
{:toc}

# Introduction

As a college student, I have to write a lot of papers with relevant citations. My usual approach to this is to dump everything out in a post-midnight binge with Netflix or Pandora on in the background, then go back through the next morning, armed with a Bibtex containing the top twenty results from Web of Science and Google, and randomly add citations after sentences based on the few words I picked up while reading the abstract. This is obviously a blatantly flippant attitude towards the dignity of the scientific process; decades of work in information retrieval should have made hand-processing of these citations totally outmoded. In this spirit, I wanted to make a tool that would take a corpus of documents and automatically insert relevant citations after each sentence.

In this blog post, I'll describe a simple information retrieval metric, TF-IDF, and how to use it to build a scalable index for retrieving documents from a large corpus. I'll mention some more advanced methods as well, in case you, the reader, want to improve on this. Finally, I'll describe how I applied this method to a speech by Donald Trump.

# Language Models

According to Wikipedia, a [language model][language-model-wiki] is "a statistical language model is a probability distribution over sequences of words". Phrased mathematically, given a sequence of words, a 
language model tells you

$$P(w_1, w_2, ..., w_m)$$

where the sentence is represented as the word sequence

$$w_1, w_2, ..., w_m$$

There has been a lot of work done about how to represent language, and many questions that should be considered when developing a good model. For example, what level of detail do you need in order to represent a language? How much more information do you get from the context of a word? To answer these questions, we can develop a language model that assumes the answer is one particular way, then see how well the model works on real-world problems.

## TF-IDF

One of the simplest language models uses a metric called Term Frequency, Inverse Document Frequency, or TF-IDF. Here are some of the assumptions the model makes:

1. The meaning of a sentence or document is adequately represented by the words it contains.
2. Context doesn't matter very much; words can be rearranged without losing too much information

TF-IDF is really a score for how much information a particular word gives us. We would like words such as "this", "we" or "and" to have a very low score, while words such as "numerical", "document" or "Trump" to have a higher score. There are two terms in TF-IDF: the first, **Term Frequency**, is how many times a term appears in a document. The second, **Inverse Document Frequency**, is the inverse of how many documents the term appears in. The two terms are multiplied together to get TF-IDF. Mathematically, this is represented as:

$$tfidf(t, d, D) = tf(t, d) \times idf(t, D)$$

where $$t$$ is the term (word) we are considering, $$d$$ is the document we are considering, and $$D$$ is the set of all documents.

The intuition for this is that we get a higher score for a term if it appears a bunch of times in a document, but the score goes down if the term appears across a bunch of documents. So the best score happens if the term appears a lot in one document but not in other documents.

We can represent **term frequency** $$tf$$ in different ways, based on further assumptions about the way language works. For this purpose, I'll represent $$tf$$ as the raw term frequency; in other words, the number of times a term appears in a document. We could normalize this by the size of the document, for example, or take the log of the term frequency.

Similarly, we can represent **inverse document frequency** $$idf$$ in different ways. For this purpose, I'll represent it using the equation

$$idf(t, D) = \log{\frac{N}{n_t}}$$

where $$N$$ is the total number of documents in the document set $$D$$, and $$n_t$$ is the number of documents that our term $$t$$ appears in.

## Example

Here is a dummy set to illustrate the above math.

|ID|Text|
|--|----|
|1|I have heard the mermaids singing, each to each. I do not think that they will sing to me.|
|2|He who sings scares away his woes.|
|3|Elvish singing is not a thing to miss, in June under the stars, not if you care for such things.|

Let's calculate the TF-IDF for the terms "sing" and "mermaid". First, let's do some simple preprocessing on this dataset. We'll say that "singing", "sings" and "sing" are basically the same, as are "mermaids" and "mermaid" (in practice, this is called lemmatization). Then we can calculate the **term frequency** for each document as:

|ID|$$tf(\text{sing},d)$$|$$tf(\text{mermaid},d)$$|
|--|---------------------|------------------------|
|1|2|1|
|2|1|0|
|3|1|0|

We can also calculate the **inverse document frequency** for each term.

|Term|$$df(t,D)$$|$$\frac{N}{n_t}$$|$$idf(t,D) = \log{\frac{N}{n_t}}$$|
|----|-----------|-----------------|----------------------------------|
|sing|3|1|0|
|mermaid|1|0.333|0.477|

Using these values, we get final TF-IDF values of:

| |1|2|3|
|-|-|-|-|
|sing|0|0|0|
|mermaid|0.477|0|0|

In this case, "sing" doesn't help us choose a document, since it shows up in all the documents, while "mermaid" does, since it only appears in one of the documents. If we were trying to select the best document for either term, this metric tells us that all the documents are equally useful (or unhelpful) for telling us about singing, but the first document is more useful than the other two for telling us about mermaids.

## TF-IDF for queries

To use TF-IDF for queries that contain multiple words, we can add the TF-IDF scores for each term. For the above example, if we were querying the phrase "mermaids singing", we would get compiled TF-IDF scores of

|ID|Score|
|--|-----|
|1|0.477|
|2|0|
|3|0|

indicating that we should choose the first document, the document with the maximum TF-IDF score. We can therefore formalize our "search engine" as

$$\text{best document}(T) = \text{argmax}_{d \in D} \sum_{t \in T}{tfidf(t, d, D)}$$

The user provides the query terms $$T = t_1, t_2, ..., t_m$$ while we provide the document set $$D$$. To use the notation we established for a language model, the TF-IDF language model defines a probability distribution

$$P(w_1, w_2, ..., w_m) \propto \sum_{d \in D} \sum_{w \in w_1, w_2, ..., w_m}{tfidf(w, d, D)}$$

## Other Language Models

The TF-IDF model is relatively simple to understand and doesn't require very much computational power to implement. However, as discussed earlier, it makes some assumptions that are questionable at best (in fact, some people probably wouldn't call it "language model"). More complex models are:

- [BM25][bm25]: An improvement on TF-IDF that deals better with unknown words, among other things
- [Latent semantic indexing (LSI)][lsi]: Also "analysis" in different contexts. Assumes that words that are close in meaning will appear in similar pieces of texts.
- [Latent Dirichlet allocation (LDA)][lda]: A "generative" language model (state-of-the-art or near state-of-the-art for many applications)

# Scaling TF-IDF to a Corpus

As mentioned earlier, one benefit of TF-IDF is that it is fast. This is because we can generate indices relatively quickly (we just need to sort them) which can be queried in constant time (where the number of words in a query is considerably smaller than the number of documents). For each term, we need to access its highest term-frequency documents, so after calculating the term-frequencies for each document, we sort them. We get a lookup table for each word that contains a list of documents, sorted in decreasing order by TF. We also maintain an IDF lookup table, which simply stores the IDF for each word.

To calculate the true IDF ratings would take linear time, as we would still have to consider the TF-IDFs for all the documents in the corpus. However, a good heuristic is to consider a constant number of documents relating to each word; since they are sorted, the first documents will be the "good" documents. If we only consider a small number of documents, it will be fast.

# Generating Citations for Sentences

If you have a bunch of documents which you want to use as citations for a paper you're writing, you can treat the sentences in your paper as query strings to find documents that might be related. TF-IDF is not ideal for these types of applications, as it throws context out the window. Ideal applications involve searching for a few unique keywords from a large corpus. However, it works moderately well.

Donald Trump is well-known for using gut-based rather than publically-available citations in his speeches, so they seemed like an ideal application. The speech in question is his [2016 RNC speech][trump-speech]. To generate a training corpus, I wrote a simple Chrome extension to quickly download Wikipedia pages, which you can get [from Github][chrome-extension].

The full annotated text can be found <a href="/resources/tfidf/tfidf_trump.html" target="_blank">here</a>. Additionally, if you want to generate your own citations, the code for this project is available [here][github-repo]. I cherry-picked some examples:

|Quotes from Donald Trump's RNC Speech|
|-------------------------------------|
|We must have the best intelligence gathering operation in the world [[china](https://en.wikipedia.org/wiki/china)]. We must abandon the failed policy of nation building and regime change that Hillary Clinton pushed in Iraq Libya Egypt and Syria [[hillary_clinton](https://en.wikipedia.org/wiki/hillary_clinton)]. Instead we must work with all of our allies who share our goal of destroying ISIS and stamping out Islamic terror [[islam](https://en.wikipedia.org/wiki/islam)].|
|Anyone who endorses violence hatred or oppression is not welcome in our country and never will be [[mexico](https://en.wikipedia.org/wiki/hillary_clinton)]. Decades of record immigration have produced lower wages and higher unemployment for our citizens especially for African American and Latino workers [[mexico](https://en.wikipedia.org/wiki/mexico)]. We are going to have an immigration system that works but one that works for the American people [[barack_obama](https://en.wikipedia.org/wiki/barack_obama)].|
|You have so much to contribute to our politics yet our laws prevent you from speaking your minds from your own pulpits [[hillary_clinton](https://en.wikipedia.org/wiki/hillary_clinton)]. An amendment pushed by Lyndon Johnson many years ago threatens religious institutions with a loss of their tax exempt status if they openly advocate their political views [[united_states_constitution](https://en.wikipedia.org/wiki/united_states_constitution)]. I am going to work very hard to repeal that language and protect free speech for all Americans [[hillary_clinton](https://en.wikipedia.org/wiki/hillary_clinton)].|

[trump-speech]: http://www.politico.com/story/2016/07/full-transcript-donald-trump-nomination-acceptance-speech-at-rnc-225974
[colbert-correspondents]: https://www.youtube.com/watch?v=2X93u3anTco
[wikicorpus]: http://www.cs.upc.edu/~nlp/wikicorpus/
[chrome-extension]: https://github.com/codekansas/citation-generator
[language-model-wiki]: https://en.wikipedia.org/wiki/Language_model
[bm25]: https://en.wikipedia.org/wiki/Okapi_BM25
[lda]: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
[lsi]: https://en.wikipedia.org/wiki/Latent_semantic_analysis
[github-repo]: https://github.com/codekansas/citation-generator
