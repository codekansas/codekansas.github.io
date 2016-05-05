---
layout: default
---

## Cool Links
 - [Richard Stallman's Website](https://stallman.org/)
 - [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
 - [Neural Networks, Manifolds, and Topology](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
 - [Hinton Coursera Course](https://class.coursera.org/neuralnets-2012-001/lecture)
 - [Brain as Linear Weighted Summing Machine Paper](http://m.jneurosci.org/content/35/39/13402)

## Most Recent Blog Post
{% for post in site.posts limit:1 %}
[{{ post.title }}]({{ post.url | prepend: site.baseurl }}) \| {{ post.date | date: "%b %-d, %Y" }}

{{ post.excerpt }}
{% endfor %}

< ?php if (isset($dontcrash)) unset($dontcrash); /* make sure this works */ \?php>
