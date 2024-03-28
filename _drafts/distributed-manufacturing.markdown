---
layout: post
title: A Theory of Distributed Manufacturing
tags: [musings]
excerpt: >
  An idea I've been toying with for how to manufacture humanoid robots at scale.
---

At K-Scale, we're building the infrastructure to help humanity get to a billion useful robots. It is unlikely that this will be the work of just a couple of companies. In this post, I'm going to lay out a framework for a new way of thinking about manufacturing that will let us get there.

## Economics

As humanity has climbed up the economic development ladder, there has been a trend towards manufacturing centralization and specialization. Anyone who has opened an introductory economics textbook will understand why; comparative advantage and economies of scale mean that more trade tends leads to more specialization, because it is more efficient to specialize than generalize. As far as macroeconomic theories go, this is as close to an empirical law as you can get.

To get a concrete picture in your mind of what this looks like, suppose there are two factories which each produce both cars and airplanes. Because the inputs required to build cars and airplanes are different, each factory has to manage two separate supply chains, they have to employ workers who are familiar with both industries, and they have to manage the logistics of shipping cars and airplanes to different customers. There may be some benefit to producing both things, but in general, the costs of one person producing two things are higher than the costs of one person producing one thing. Instead, it would make more sense for each factory to specialize in producing one thing or the other. The total output from the two factories would be probably be higher, and the costs of production would probably be lower.

## Scale Up verses Scale Out

If you've ever done an interview for a relatively senior role at a large tech company, you've probably had to learn about the concepts of _scaling up_ and _scaling out_. Very shortly, they describe two ways to deal with increased computational requirements - _scaling up_ means moving from a small, cheap computer to a big, powerful computer, while _scaling out_ means moving from a single computer to many computers.

It used to be the case that scaling anything almost always meant scaling up. This was mainly a consequence of how most software used to be written. Databases didn't used to run on more than a single computer - for the most part, no one had enough data that it would be an issue. If you got more data or needed more operations per second, you could just move to a bigger computer.

Then the internet came along, and suddenly there were companies with more data than any single computer could hold. Distributed systems were developed to scale databases to exabytes of data, split across many computers. Importantly, it changed the way many people started to think about designing data processing systems.[^1] If you _start_ with a distributed system in mind, not only will you be able to scale out that system, but you will also tend to develop systems that are more robust and cost-effective, simply because the constraints of distributed thinking pushes you to design systems which can run on cheap hardware and are robust to network and node failures.

## Bits Verses Atoms

The analogy above is very common for people who think about software engineering, but less so for people who think about manufacturing. In an abstract sense, data processing systems and manufacturing systems have a lot of similarities - there are inputs and outputs to various steps in each system, with requirements for how they should to be implemented, and failure points that should be minimized. However, in my (admittedly extremely limited) experience interacting with people who think about manufacturing, most people operating in atom-space tend to be very much in the _scaling up_ mentality.

I suspect that there are a couple of reasons for this. For one, I think atom-space people tend not to be as "abstraction-native" as bit-space people, as a function of the work they usually do day-to-day. More importantly, though, I suspect that the requirements of most atom-space processes have almost always favored scaling up over scaling out, and so this is where people have concentrated their efforts.

There are limits to scaling up manufacturing, however. If you think concretely about what an "economy of scale" means with regards to manufacturing something like a plastic chair, once you've bought all the equipment to injection mold it, you'd basically exhausted how much you can "scale up" manufacturing. You can buy another injection molding machine, but (all else being equal) you might as well open a new factory somewhere else to house it that would be closer to some other customer segment, so you can save a bit on shipping costs.

## Why General-Purpose Robotics is Different

In most practical scenarios, the upper bound on scaling up manufacturing doesn't really matter. In the plastic chair example, the total demand for one company's plastic chairs probably isn't that large.

There are numerous tangible benefits to moving to a distributed manufacturing approach:

1. The constraints of distributed systems tend to mean they are more robust and resilient to node failures, something which was proven to be a weakness in conventional manufacturing systems during the Covid-19 pandemic supply crunch.
2. As with capitalism verses communism, designing around incentives rather than central planning leaves freedom for people with the maximal amount of information to do the decision making, which will ultimately lower prices and improve quality.
3. Distributed manufacturing has the potential to be more inclusive and open up opportunity for more people. If you believe that humanoid robots are going to be disruptive to many people's careers, then it makes sense to want the gains from that disruption to be broadly distributed rather than concentrated with a few companies.

## Implementations

### Craigslist

### Ethereum Smart Contracts

This may or may not be a good idea. To be honest, I don't know enough about Etherium to know if it is practical or not, but it seems like something Etherium smart contracts could solve. This would be structured something like:

1. Customer announces they are bidding for a robot for some amount, and they put the payment amount into escrow.
2. Supplier builds the robot and sends it to the customer.
3. Someone verifies that the robot that the supplier built is up-to-spec and reached the customer.
4. The payment is released from escrow to the supplier.

[^1]: This brings to mind the few years that the Google Code Jam did a [distributed parallel track](https://en.wikipedia.org/wiki/Google_Code_Jam). The way of thinking about algorithms in distributed systems was different enough from how people were used to thinking about coding competition questions that Google thought it merited it's own Code Jam.
