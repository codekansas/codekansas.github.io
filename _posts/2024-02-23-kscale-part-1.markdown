---
layout: post
title: Humanoid Robots are a Low Capex Problem with Favorable Unit Economics
tags: [kscale]
excerpt: >
  Part one of a series of thoughts about humanoid robots.
---

In this post, I will argue for two points that I think are slightly contrarian. These points are:

1. Humanoid robots are a low capex problem.
2. Humanoid robots have favorable unit economics.

## Humanoid robots are a low capex problem

The conventional wisdom in Silicon Valley is that hardware is high capex. The thinking goes that, for a hardware company, once you demonstrate something works and is profitable, if you are not well-funded then a better-funded competitor will just come along and copy it (Pebble being the canonical example). Therefore, the only way to win in this space is to raise enough money to outspend everyone else.

With regards to humanoid robots, this thinking is basically wrong.

### Lessons from a related industry

Rather than thinking about humanoid robot companies as hardware companies, the better analogy to use is the self-driving car industry. I've been around machine learning long enough to remember a time when everyone thought self-driving cars were relatively easy and close to being solved (eight years ago). As a company backed by Y Combinator, I'm going to primarily reference other Y Combinator companies as I think their founder profile is likely more similar to mine, but I think the lessons are transferrable to other companies as well. So here are a few Y Combinator-backed companies that raised money on this premise:

- Cruise
- Embark
- Starsky
- May Mobility
- Faction

Not exactly a list of roaring successes, although you'd be hard-pressed to find any companies that has really succeeded in the self-driving space.

So what went wrong? I got to talk to a couple founders of these companies, and the number one thing they told me was to **keep costs low**. Apparently spending a bunch of money on hardware means you run out of money pretty quickly.

At the time, of course, a lot of people imagined the self-driving problem was a lot closer to being solved than it turned out to be. On the heels of the DARPA Grand Challenge, it seemed like tweaking a few things here and there would be all it took to get a self-driving car operating fully autonomously on the road. What actually happened, of course, was that we had to rethink the problem from the ground-up; the most promising approaches today are based on end-to-end self-supervised learning, a far cry from LIDAR, HD maps and bounding boxes.

Basically, businesses need to make more money than they spend. For self-driving, this meant either doing the Elon-type bait-and-switch (something that is actually pretty hard to pull off), or delivering on an insane level of engineering execution (which is also pretty hard to pull off).

Fortunately, "insane level of engineering execution" is not a high capex problem. Actually, the best engineering teams I know are all relatively low capex, they're just really good at what they do.

### Applying those lessons to humanoid robots

It feels like we're in a very similar moment for humanoid robots. There are specifically at least two broad areas in which I think the field has a high likelihood of dying in the same way.

#### Funding

It seems obvious to a lot of people that if we just take the techniques that have proven to work well for language models and make them multi-modal, then we should get a really useful humanoid robot. This has prompted a number of people who don't have much experience in machine learning to jump onto the hype train (interestingly, Tim Kentley Klay is going for a repeat here).

Personally, as someone who is interested in actually seeing humanoid robots come to fruition and not just cashing out, this makes me really worried about the same dynamic playing out, where a few large companies raise so much money that they suck the air out of the room, fail to deliver, and funding dries up by the time we've figured out how to actually solve the problem.

For self-driving cars, this fundamentally came down to a misread of the situation in 2016. Many people were under the impression that the thing that would kill an investment was failure to reach critical mass, so they invested in companies that seemed like they could raise the kind of money required to do that.

In fact, the thing that killed most of the investments was that the problem was a lot harder than people thought. In this thinking, the right thing to have done would have been to keep costs low, build a top-notch research team, and wait for the moment to strike.

Ironically, this means that for a company like Cruise, there's an argument to be made that a few marginal dollars would have been better spent investing in competitors than in the company itself. This would have helped to fight against the dogmatic thinking that tends to arise inside companies - a cottage industry of self-driving car companies reinventing the wheel would have probably come up with better solutions earlier on.

#### Engineering

Having spent time working on self-driving at Tesla, an important less is that **it takes many things simultaneously being executed correctly to solve the problem, and poor execution in just one area can kill everything**. Practically speaking, this means that the only companies which have a chance at succeeding are the ones lead by people who understand the entire stack on a good-enough level that they can call bullshit in the right places, and have clear visions to execute on. Even then, there's a good chance the vision might be wrong.

At present, the only company that I think has the right pieces in place is 1X, and they probably do the least amount of marketing of any of the companies in the space.

## Humanoid robots have favorable unit economics

In conversations with intelligent people about how to launch a humanoid robot company, it has been suggested to me that I follow the Tesla model. Essentially, the idea is that if you first make a really great product that appeals to a niche market of high-end buyers, like the Tesla Roadster, you can then expand into the rest of the market as you increase production and lower costs, as with the Tesla Model S and subsequently the Model 3. Some other products from other companies which have followed this model are the iPhone, the Oculus Rift, the Dyson vacuum, and the Peloton bike.

The reasoning for this is pretty easy to follow. You should make a great product with high margins before you've figured out how to bring costs down, because if you don't, you'll make a bad product with low margins and your company will die.

With regards to humanoid robots, this thinking is also basically wrong.

{% include kscale.html %}
