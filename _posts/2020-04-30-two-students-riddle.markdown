---
layout: post
title: "Two Students Riddle"
category: 🧩
excerpt: >
  There is a teacher and 2 students in a classroom. The students are A and B. The teacher thinks of 2 positive integers and tells the sum of those numbers to student A without student B hearing it. Then tells their product to student B without student A hearing it. After this, the teacher asks the 2 students what was the 2 numbers. First student A says: I don't know. Then student B says: I don't know either. After hearing this, student A says: Now I know. Then student B says: Now I know them too. What were the 2 numbers?
math: true
---

A friend of mine recently sent me the following math problem.

> There is a teacher and 2 students in a classroom. The students are A and B. The teacher thinks of 2 positive integers and tells the sum of those numbers to student A without student B hearing it. Then tells their product to student B without student A hearing it. After this, the teacher asks the 2 students what was the 2 numbers.
>
> First student A says: I don't know.
>
> Then student B says: I don't know either.
>
> After hearing this, student A says: Now I know.
>
> Then student B says: Now I know them too.
>
> What were the 2 numbers?

As an extension:

> What if the two numbers must be unique? That is, both students know the numbers must be unique?

If you don't want to have the answer spoiled (or at least, my version of the answer), don't read the next paragraph! I'm going to write some about how I approached it. Here is a diagram of the problem statement.

{% include /images/riddles/two_students.svg %}

------

# Solution

This can be broken down into four steps. Let's give student A the number $P$ and student B the number $Q$.

1. Student A can't figure out the two numbers. This means that $P$ must have more than one additive decomposition $P = x + y$, where $x$ and $y$ are positive integers. Fortunately almost every positive integer satisfies this property, except for `1` (which has no additive decompositions) and `2` and `3` (which have exactly one each, `1 + 1` and `1 + 2`).
2. Student B can't figure out the two numbers as well. This means that $Q$ must have more than one multiplicative decomposition $Q = x y$. This means that $Q$ cannot be a prime number; otherwise, student B would be able to figure out the answer.
3. After hearing that Student B can't figure out the number either, Student A realizes that the number must not be prime. This means that Student A can rule out any additive decompositions of $P$ which consist of a prime number plus 1. Since they are able to figure it out afterwards, there must be exactly one pair left over.
4. After hearing that Student A has figured out the answer, Student B can rule out any additive decomposition which wouldn't have been ruled out by the second step.

I am pretty lazy and a mediocre mathematician, so rather than trying to figure it out mathematically, I decided to figure it out through code. I also think putting it in code makes it a bit clearer. So let's code up each step and test out some numbers!

# Program

Let's define a helper function for creating a unique tuple from two numbers. This will ensure that we don't accidentally double-count pairs.

{% highlight python %}
def _tup(x: int, y: int) -> Tuple[int, int]:
    return (x, y) if x < y else (y, x)
{% endhighlight %}

## Additive Decompositions

We can use the following function to find all of the additive decompositions for some number.

{% highlight python %}
def add_decomp(s: int) -> Set[Tuple[int, int]]:
    return {_tup(i, s - i) for i in range(1, s // 2 + 1)}
{% endhighlight %}

## Multiplicative Decomposition

We can use the following function to find all of the multiplicative decompositions for some number.

{% highlight python %}
def prod_decomp(p: int) -> Set[Tuple[int, int]]:
    return {_tup(i, p // i) for i in range(1, int(sqrt(p)) + 1) if p % i == 0}
{% endhighlight %}

## Step 1

For Student A to be unable to figure out the first number, there must be at least two additive decompositions.

{% highlight python %}
def a_first(s: int) -> bool:
    return len(add_decomp(s)) > 1
{% endhighlight %}

## Step 2

For Student B to also be unable to figure out the first number, there must be at least two multiplicative decompositions (which also pass the first step).

{% highlight python %}
def b_first(p: int) -> bool:
    return sum([a_first(i + j) for i, j in prod_decomp(p)]) > 1
{% endhighlight %}

## Step 3

After Student B has announced they are unable to figure out the answer either, Student A can rule out any pair which would have let Student B figure out the answer.

{% highlight python %}
def a_second(s: int) -> bool:
    return sum([b_first(i * j) for i, j in add_decomp(s)]) == 1
{% endhighlight %}

## Step 4

Finally, now that Student A has announced they are able to figure out the answer, Student B can rule out any pair which would not have let Student A figure out the answer.

{% highlight python %}
def b_second(p: int) -> bool:
    return sum([a_second(i + j) for i, j in prod_decomp(p)]) == 1
{% endhighlight %}

## Memoization

Notice that a lot of these functions are called multiple times with the same inputs. We can *memoize* the function calls, since they always return the same value, using the following function.

{% highlight python %}
def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:            
            memo[x] = f(x)
        return memo[x]
    return helper
{% endhighlight %}

## The Whole Program

We can string all of these functions together and search through all the possible pairs from 1 to 5.

{% highlight python %}
from math import sqrt
from typing import Tuple, Set

def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:            
            memo[x] = f(x)
        return memo[x]
    return helper

def _tup(x: int, y: int) -> Tuple[int, int]:
    return (x, y) if x < y else (y, x)

@memoize
def add_decomp(s: int) -> Set[Tuple[int, int]]:
    return {_tup(i, s - i) for i in range(1, s // 2 + 1)}

@memoize
def prod_decomp(p: int) -> Set[Tuple[int, int]]:
    return {_tup(i, p // i) for i in range(1, int(sqrt(p)) + 1) if p % i == 0}

@memoize
def a_first(s: int) -> bool:
    return len(add_decomp(s)) > 1

@memoize
def b_first(p: int) -> bool:
    return sum([a_first(i + j) for i, j in prod_decomp(p)]) > 1

@memoize
def a_second(s: int) -> bool:
    return sum([b_first(i * j) for i, j in add_decomp(s)]) == 1

@memoize
def b_second(p: int) -> bool:
    return sum([a_second(i + j) for i, j in prod_decomp(p)]) == 1

def log(x: int, y: int, s: str) -> None:
    print(f'x: {x}, y: {y} :: {s}')

def test(x: int, y: int, do_log: bool = False) -> bool:
    s, p = x + y, x * y
    if not a_first(p):
        if do_log:
            log(x, y, 'a knows the first time')
        return False
    if not b_first(s):
        if do_log:
            log(x, y, 'b knows the first time')
        return False
    if not a_second(p):
        if do_log:
            log(x, y, 'a does not know the second time')
        return False
    if not b_second(s):
        if do_log:
            log(x, y, 'b does not know the second time')
        return False
    return True

for x in range(1, 6):
    for y in range(1, x + 1):
        if test(x, y, do_log=True):
            print(f'{x} and {y} works!')
{% endhighlight %}

This gives the results for all the pairs from 1 to 5:

{% highlight bash %}
x: 1, y: 1 :: a knows the first time
x: 2, y: 1 :: a knows the first time
2 and 2 works!
x: 3, y: 1 :: a knows the first time
x: 3, y: 2 :: b knows the first time
x: 3, y: 3 :: a does not know the second time
x: 4, y: 1 :: b knows the first time
x: 4, y: 2 :: a does not know the second time
x: 4, y: 3 :: b knows the first time
x: 4, y: 4 :: a does not know the second time
x: 5, y: 1 :: a does not know the second time
x: 5, y: 2 :: b knows the first time
x: 5, y: 3 :: a does not know the second time
x: 5, y: 4 :: a does not know the second time
x: 5, y: 5 :: a does not know the second time
{% endhighlight %}

# Extension

What if the two numbers must be unique? That is, both students know the numbers must be unique?

This can be done by modifying two small parts of the program.

{% highlight python %}
@memoize
def add_decomp(s: int) -> Set[Tuple[int, int]]:
    return {_tup(i, s - i) for i in range(1, s) if i != s - i}

@memoize
def prod_decomp(p: int) -> Set[Tuple[int, int]]:
    return {_tup(i, p // i) for i in range(1, int(sqrt(p)) + 1) if p % i == 0 and i != p // i}
{% endhighlight %}

This gives the results for all the pairs from 1 to 5:

{% highlight bash %}
x: 1, y: 1 :: a knows the first time
x: 2, y: 1 :: a knows the first time
x: 2, y: 2 :: a knows the first time
x: 3, y: 1 :: a knows the first time
3 and 2 works!
x: 3, y: 3 :: b knows the first time
x: 4, y: 1 :: b knows the first time
4 and 2 works!
x: 4, y: 3 :: a does not know the second time
x: 4, y: 4 :: a does not know the second time
x: 5, y: 1 :: b knows the first time
x: 5, y: 2 :: a does not know the second time
x: 5, y: 3 :: a does not know the second time
x: 5, y: 4 :: a does not know the second time
x: 5, y: 5 :: b knows the first time
{% endhighlight %}
