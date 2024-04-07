---
layout: post
title: A Theory of Distributed Manufacturing
tags: [musings]
excerpt: >
  An idea I've been toying with for how to manufacture humanoid robots at scale.
---

At K-Scale, we're building the infrastructure to help humanity get to a billion useful robots. We believe this will be fundamentally different from any other product ever made. In this post, I'm going to lay out a framework for a new way of thinking about manufacturing that I think will get us there.

## Scale Up verses Scale Out

If you've ever done an interview for a relatively senior role at a large tech company, you've probably had to learn about the concepts of _scaling up_ and _scaling out_. Very shortly, they describe two ways to deal with increased computational requirements - _scaling up_ means moving from a small, cheap computer to a big, powerful computer, while _scaling out_ means moving from a single computer to many computers.

It used to be the case that scaling anything almost always meant scaling up. This was mainly a consequence of how most software used to be written. Databases didn't used to run on more than a single computer - for the most part, no one had enough data that it would be an issue. If you got more data or needed more operations per second, you could just move to a bigger computer.

Then the internet came along, and suddenly there were companies with more data than any single computer could hold. Distributed systems were developed to scale databases to exabytes of data, split across many computers. Importantly, it changed the way many people started to think about designing data processing systems.[^1] If you _start_ with a distributed system in mind, not only will you be able to scale out that system, but you will also tend to develop systems that are more robust and cost-effective, simply because the constraints of distributed thinking pushes you to design systems which can run on cheap hardware and are robust to network and node failures.

## Bits Verses Atoms

The analogy above is very common for people who think about software engineering, but less so for people who think about manufacturing. In an abstract sense, data processing systems and manufacturing systems have a lot of similarities - there are inputs and outputs to various steps in each system, with requirements for how they should to be implemented, and failure points that should be minimized. However, in my (admittedly limited) experience interacting with people who think about manufacturing, most people operating in atom-space tend to be very much in the _scaling up_ mentality.

I suspect that there are a couple of reasons for this. For one, I think atom-space people tend not to be as "abstraction-native" as bit-space people, as a function of the work they usually do day-to-day. More importantly, though, I suspect that the requirements of most atom-space processes have almost always favored scaling up over scaling out, and so this is where people have concentrated their efforts.

There are limits to scaling up manufacturing, however. If you think concretely about what an "economy of scale" means with regards to manufacturing something like a plastic chair, once you've bought all the equipment to injection mold it, you'd basically exhausted how much you can "scale up" manufacturing. You can buy another injection molding machine, but (all else being equal) you might as well open a new factory somewhere else to house it that would be closer to some other customer segment, so you can save a bit on shipping costs.

## Why General-Purpose Robotics is Different

In most practical scenarios, the upper bound on scaling up manufacturing doesn't really matter. In the plastic chair example, the total demand for one company's plastic chairs probably isn't that large. I hope that this will not be the case for general-purpose humanoid robots. I suspect that at some capability level, the demand curve for humanoids is close to perfectly inelastic. I'm not the only person that thinks this; it is why most humanoid robot companies are building robots with six (or even seven) figure price tags. Once they make them work well, basically any price will be justifiable.

The conundrum is that most of the first robots will be basically useless. There is a cold start problem to humanoid robots that doesn't exist with other products at similar price points, like cars. There are relatively successful companies that build less than a hundred cars a year - if anything, the exclusivity is part of what makes them valuable.

The problem is that humanoid robots are a lot closer to self-driving cars than to luxury car manufacturing. The critical thing to get right with any machine learning product is the data feedback loop. The right mental image for this is the way that Midjourney presents you with four images, and you can choose which one you want to upscale - that signal tells Midjourney which images it's users tend to prefer. At scale, this feedback gives Midjourney a huge advantage over other image generation products, and it's a big part of why it's been so sticky. There are bespoke image generation models tailored to specific categories of images like cat pictures, but they are basically all worse than Midjourney.

The analogy carries well to humanoid robots. There is no "Tesla Roadster" equivalent for humanoid robots because you _need_ the scale to build something worth buying, and the only way to get to scale when you start with a bad product is by selling it very cheaply. In this sense, the right go-to-market strategy for humanoid robots probably more closely resembles the PlayStation or another loss leader. Unfortunately, that strategy is not possible if getting each robot out the door costs $100,000, but it might be possible if your robot costs about as much as an Apple Vision Pro.

## Implementations

I have a couple different ideas on how to facilitate "scale out" manufacturing.

### Spiderweb

This is the simple approach of just building many small factories that each produce humanoids. I expect spinning up a single factory to cost about $10 million and be capable of producing something like 100 robots per year at the start.

### Craigslist

### Ethereum Smart Contracts

This may or may not be a good idea. To be honest, I don't know enough about Etherium to know if it is practical or not, but it seems like something Etherium smart contracts could solve. This would be structured something like:

1. Customer announces they are bidding for a robot for some amount, and they put the payment amount into escrow.
2. Supplier builds the robot and sends it to the customer.
3. Someone verifies that the robot that the supplier built is up-to-spec and reached the customer.
4. The payment is released from escrow to the supplier.

This feels like a good application of crypto because it can help stimulate demand before a physical product is built, without keeping a bunch of cash in escrow in a single location. Also, crypto enthusiasts as a category of consumers seems more likely than average to want to purchase a humanoid robot.

[^1]: This brings to mind the few years that the Google Code Jam did a [distributed parallel track](https://en.wikipedia.org/wiki/Google_Code_Jam). The way of thinking about algorithms in distributed systems was different enough from how people were used to thinking about coding competition questions that Google thought it merited it's own Code Jam.
