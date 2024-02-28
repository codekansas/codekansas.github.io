---
layout: post
title: Humanoid Robots are a Low Capex Problem
tags: [kscale]
excerpt: >
  Part one of a series of thoughts about humanoid robots.
---

This is the first in a series of posts describing my ideas around humanoid robots. These ideas are:

1. Humanoid robots are a low capex problem.
2. Humanoid robots have favorable unit economics.
3. Humanoid robots are an important to work on.

This post will be about the first point.

## Conventional Wisdom

In broad strokes, the conventional wisdom in Silicon Valley is that hardware is high capex. The thinking goes that, for a hardware company, once you demonstrate something works and is profitable, if you are not well-funded then a better-funded competitor will just come along and copy it (Pebble being a canonical example). Therefore, the only way to win in this space is to raise enough money to outspend everyone else, and get the timing right.

{% katexmm %}
From a different angle, humanoid robots seem like a risky long-term research bet which won't turn into a sustainable business for five to ten years, and will require large up-front investments before yielding sustained returns. In DCF terms, we can say that $CF_{1 \cdots 5}$ are all large negative numbers, meaning that in a non-zero interest rate environment, we can only make this bet if our $CF_{6 \cdots \infty}$ are very large positive numbers.
{% endkatexmm %}

With regards to humanoid robots, I would say that this thinking is basically wrong.

## Lessons from a related industry

First, I should say that I personally struggle to think of any historical examples where betting large amounts of money on a few people to solve hard problems has actually worked out. There's an information gap between people making investments and people allocating those resources. I think most VCs are, in fact, pretty savvy to this, but then there's also a tendency to think that if you can just suck all the air out of the room with some large bet into a single company then that company will be guaranteed to win because it will aggregate all the good ideas into one place.

Anyway. If you view the humanoid robot industry and narrative around it through the lens of the assumptions I described earlier, a lot of the way people are approaching the problem will start to make more sense. From where I sit as someone interested in seeing humanoid robots become a reality sooner rather than later, I find this worrying because of how closely it seems to parallel another much-lauded machine learning industry from several years ago - self-driving cars and trucks.

I've been around machine learning long enough to remember a time when everyone thought self-driving cars were relatively easy and close to being solved (eight years ago). The DARPA Grand Challenge had just happened and it seemed like a few tweaks here and there were all that was needed to make the next trillion-dollar company.

As a company backed by Y Combinator, I'm going to primarily reference other Y Combinator companies as I think their founder profile is likely more similar to mine, but I think the lessons are transferrable to other companies as well. So here are a few Y Combinator-backed companies that raised money on this premise:

- Cruise
- Embark
- Starsky
- May Mobility
- Faction

Not exactly a list of roaring successes, although you'd be hard-pressed to find any companies that has really succeeded in the self-driving space.

So what went wrong? I got to talk to a couple founders of these companies, and the number one thing they told me was to **keep costs low**. Apparently spending a bunch of money without concrete revenue means you run out of money pretty quickly.

Today, it seems like we are actually getting close to having fully-autonomous taxis. Personally, when I visit San Francisco, Waymo has completely replaced Uber. What actually happened, though, was that we had to rethink the problem from the ground-up; the most promising approaches today are based on end-to-end self-supervised learning, a far cry from LIDAR, HD maps and bounding boxes.

At first glance, this seems like it should validate the high capex thesis - Waymo was the only company that could keep burning money until they figured out the right approach. But I think this is the wrong take-away. I think the correct take-away is that a large number of people made huge bets on the wrong approach, and when they failed it seemingly confirmed to investors that the problem was a waste of money. Waymo was the only company that fostered an internal culture that allowed the right approach to eventually emerge.

### Why did so many people make the same mistake?

There is a counterfactual universe where a machine learning researcher realizes that model scaling laws are really impressive, and rather than trying to hack together a demo of getting a car to drive, they start investigating self-supervised methods for end-to-end learning. They figure out how to learn world models from video data, validating those models in simulation or on cheap hardware (like RC cars), and by the time they're spending real money to build a full-sized car they've got a system that's actually pretty close to working.

In this universe, I think there's basically zero chance of that person's company getting funded. It was not clear to many people in 2016 that self-supervised learning would be the roaring success it turned out to be, and I suspect that bringing a self-driving RC car in front of some investors would get you laughed out of the room when any "serious" self-driving company could actually take you for a ride around their parking lot in an autonomous taxi.

Additionally, deep learning itself is not amenable to great one-off demos. If you have a thousand images worth of data, kernel methods work better than deep neural networks.

## Not repeating the same mistakes with humanoid robots

It is clear in hindsight that just because a car can drive on a closed course in the desert doesn't mean it can drive in San Francisco. Similarly, just because a humanoid robot can make coffee doesn't mean that we've reached the "ChatGPT moment" for robots.

There are specifically at least two broad areas in which I think the field has a high likelihood of dying in the same way.

### Misallocated Resources

At present, it seems "obvious" to some people that if we just take the techniques that have proven to work well for language models and make them multi-modal, then we should get a really useful humanoid robot. This has prompted a number of people who don't have much experience in machine learning to jump onto the hype train (interestingly, Tim Kentley Klay is going for a repeat).

Personally, as someone who is interested in actually seeing humanoid robots come to fruition and not just cashing out, this makes me really worried about the same dynamic playing out, where a few large companies raise a lot of money on subpar engineering, suck the air out of the room, fail to deliver, and funding dries up by the time we've figured out how to actually solve the problem.

For self-driving cars, this fundamentally came down to a misread of the situation in 2016. Many people were under the impression that the thing that would kill an investment was failure to reach critical mass, so they invested in companies that seemed like they could spend their way to that point. In fact, the thing that killed most of the investments was betting on the wrong technical approach and spending too much money too quickly.

### Engineering Bloat

Okay, maybe you buy that most of the self driving companies were taking the wrong approach. But people learned their lesson, and now they're going to do it the right way. Maybe we don't have an exact path to a humanoid robot, but a big, well-funded organizations can act more like research labs today, trying out different approaches the way that Waymo did. Money is cheaper today and people are smarter about keeping capex low. What's wrong with that?

The answer to this question is actually the exact reason that I decided to leave the robotics research lab at Meta to work on a startup, and a big part of why I think Tesla's self-driving efforts are not succeeding, despite having all of the right pieces in place:

**In order to execute on an idea, you have to understand the idea completely.**

Most of machine learning, and robotics in particular, is riddled with all these small, practical details that are critical to the success of a system - a single thing done incorrectly can can kill the entire thing. Most of the time spent figuring out how to make a machine learning model work is spent on ironing out these small details. The difference between a good machine learning engineer and a bad one basically comes down to an intuitive sense of what details matter and how to do them correctly.

This is why robotics research is actually very difficult, because the skillsets required to train models and the skillsets required to build robots are pretty divergent, but the best work is done at their intersection. Look at the most exciting results from the last few months:

- [Universal Manipulation Interface](https://arxiv.org/abs/2402.10329)
- [Mobile Aloha](https://mobile-aloha.github.io/)
- [OK Robot](https://arxiv.org/abs/2401.12202)

What do these papers have in common? They're all done by working throughout the full robotics stack. The people training the models made at least some part of the CAD model for the robot as well. At one point, Mahi (one of the authors on the OK Robot paper) literally sent my team the model for the iPhone mount they designed for the end of the Stretch to make it work.

Practically speaking, this means there is very little return to growing a machine learning team beyond the point that a single person could work throughout the entire stack without having to go through anyone else. If you want proof, just ask yourself why Tesla FSD still isn't fully autonomous.

### Shipping Slowly

If you were to poll machine learning researchers and ask what the most exciting meta-trend in the field has been in the last five years, I think a decent number would say the move from research to production. I personally was shocked the first time I heard an AI-generated [Kayne singing Hey There, Delilah](https://www.youtube.com/watch?v=-9Ado8D3A-w) and realized it was using a model from a paper that I had coauthored. It confirmed to me that the field is entering a mature phase where research is actually doing useful real-world things.

There is starting to emerge a playbook for building great machine learning products. It looks something like this:

1. Release a bad but compelling model for some task like image, video, audio, or text generation
2. Build a feedback look from that model which helps it get better
3. Continually ship model improvements

Some examples of companies that have done this well are:

- Midjourney
- Suno
- Character
- Pika

At present, we're in a phase where many people consider robots to be confined to the lab. Even the most ambitious robotics companies are reluctant to move into the real world. This tracks closely with the self-driving car industry's reluctance to deal with the hard problem of real-world navigation, instead focusing on highly controlled environments.

Successful machine learning models can be described roughly as "solve everything, then ask it to do your thing". This is as opposed to the approach of "solve one task, then solve more and more tasks". If Midjourney had started out as a cat picture generator, it would not have been very good. The fact that it can generate compelling pictures of cats today is because it tried to solve everything at once. This is a side-effect of intelligence being very transferable.

Success in the robotics industry will ultimately be determined by the company that can ship a compelling general-purpose robot quickly, and iterate on that robot to make it better. Doing that is a hard, unsolved problem, but it is the only way that works in machine learning.

## A Model for a Good Humanoid Robot Company

At present, I think there is basically only one company that has a chance of making a useful humanoid robot.

![1X](/images/kscale/1x.webp)

Oddly enough, they are also the least active on social media. From what I've heard their CEO has a very strong mindset of "let the product speak for itself" - because he's actually a great engineer who cares about the success of the space, and doesn't want to prematurely overhype something and kill it for everyone else.

I think there is space for another company. In ten years the industry will either be worth multiple trillions of dollars or be worth zero, and I suspect that having multiple good companies instead of one good company and a bunch of scammers sucking air out of the room will help it track towards the former.

So, what does that company look like? Here's some ingredients:

1. Small: The whole team understands the whole stack
2. Ship something cheap quickly: The first product should be more like a very sophisticated toy than an industrial robot
3. Consumer-focused: Getting your humanoid to work on an assembly line is basically useless and a distraction from the actually valuable problem of real-world interaction

I think this type of team maps a lot more closely to how software companies are typically organized than how hardware companies are organized.

Anyway, hopefully after reading this you buy that humanoid robotics is low capex. In the next post, I'll argue why I think good execution will deliver software-style unit economics.

{% include kscale.html %}
