---
layout: post
title: "Combination Lock Riddle"
category: 🧩
excerpt: >
  A combination for a lock has 3 wheels, X, Y, and Z, each of which can be set to eight different positions. The lock is broken and when any two wheels of the lock are in the correct position, the lock will open. Thus, anyone can open the lock after 64 tries (let A and B run through all possible permutations). However, the safe can be opened in fewer tries! What is the minimum number of tries that can be guaranteed to open the lock?
math: true
---

A friend of mine recently posed the following math problem.

> A combination for a lock has 3 wheels, X, Y, and Z, each of which can be set to eight different positions. The lock is broken and when any two wheels of the lock are in the correct position, the lock will open. Thus, anyone can open the lock after 64 tries (let A and B run through all possible permutations). However, the safe can be opened in fewer tries! What is the minimum number of tries that can be guaranteed to open the lock?

This is actually a really neat problem, and the solution feels somewhat non-intuitive. If you don't want to see the answer yet, don't read below.

------

## Solution

Ready? With the right search strategy, you would only have to try 32 combinations (this is as low as I could make it, at least). I think there is a really cool 3D interpretation that can be used here.

If you think about the search space as an `8x8x8` cube, then there are three intersecting columns which represent the answers, where `(X, Y)` is correct, `(X, Z)` is correct, or `(Y, Z)` is correct. I whipped this visualization up in OpenSCAD, representing a lock where the correct combination is `X = 3, Y = 5, Z = 2`.

<iframe id="vs_iframe" src="https://www.viewstl.com/?embedded&url=https%3A%2F%2Fben.bolte.cc%2Fimages%2Friddles%2Fcombo.stl"></iframe>

Finding a solution to this riddle means finding some manifold which intersects any possible set of columns. This is pretty easy to do with 64 guesses, since you can just cover one wall. Is it possible to do it in fewer?

Intuitively, if we're trying to get fewer than 64 guesses, it makes sense to try to have some kind of triangle cutting the cube. However, it is hard to cover the entire space with just one triangle. One approach is to try to cover opposite corners separately; the back upper right corner and the front lower left corner, for example. If we can cover these corners completely, then at least one of the columns will have to intersect at least one of our guesses.

By tinkering around in OpenSCAD I was able to figure out a reasonable solution. Here is my code:

{% highlight scad %}
module dot(x, y, z) {
    color([1, 1, 1, 0.4])
        translate([x, y, z])
            cube([1, 1, 1]);
    echo(x, y, z);
}

module sol(N, X, Y, Z) {
    M = floor(N / 2);
    for (i = [1 : M])
        for (j = [1 : M])
            dot(i, j, (j + i - 1) % M + 1);
    for (i = [1 : M])
        for (j = [1 : M])
            dot(M + i, M + j, M + (j + i - 1) % M + 1);
}

manifold(8);
{% endhighlight %}

As an illustration, the diagram below shows this manifold if each wheel were given 32 possible values. If you rotate it around, maybe you can see how this configuration covers the two corners completely. Also, it forms four nice triangles in 3D space.

<iframe id="vs_iframe" src="https://www.viewstl.com/?embedded&url=https%3A%2F%2Fben.bolte.cc%2Fimages%2Friddles%2Fmanifold.stl"></iframe>

For the original problem statement, here is the complete list of guesses that we would have to make to be sure of unlocking it in 32 steps.

{% highlight bash %}
ECHO: 1, 1, 2
ECHO: 1, 2, 3
ECHO: 1, 3, 4
ECHO: 1, 4, 1
ECHO: 2, 1, 3
ECHO: 2, 2, 4
ECHO: 2, 3, 1
ECHO: 2, 4, 2
ECHO: 3, 1, 4
ECHO: 3, 2, 1
ECHO: 3, 3, 2
ECHO: 3, 4, 3
ECHO: 4, 1, 1
ECHO: 4, 2, 2
ECHO: 4, 3, 3
ECHO: 4, 4, 4
ECHO: 5, 5, 6
ECHO: 5, 6, 7
ECHO: 5, 7, 8
ECHO: 5, 8, 5
ECHO: 6, 5, 7
ECHO: 6, 6, 8
ECHO: 6, 7, 5
ECHO: 6, 8, 6
ECHO: 7, 5, 8
ECHO: 7, 6, 5
ECHO: 7, 7, 6
ECHO: 7, 8, 7
ECHO: 8, 5, 5
ECHO: 8, 6, 6
ECHO: 8, 7, 7
ECHO: 8, 8, 8
{% endhighlight %}
