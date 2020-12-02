---
layout: post
title: "The Work-Procrastinate Cycle"
category: ☀️
excerpt: The results of an experiment I've been conducting on myself.
---

For a long time, I've struggled with procrastination. Unfortunately, as someone who is filled with what a friend once called "undirected ambition," my procrastination usually takes the form of periods of high-energy productivity followed by periods of low-energy decompression.[^crab-ganglion] For what it's worth, from what I understand this isn't particularly unusual, although it manifests itself in different forms.[^procrastination] Several months ago, I read a Hacker News comment about someone who mentored junior developers, and their first piece of advice was to track the amount of time you spend doing various activities throughout the day. Some people he mentored realized they were spending 5 to 6 hours a day on Reddit, Facebook, Hacker News or other social media.

## Data

I thought this was an interesting experiment, so I decided to do it on myself.[^methods] I added [this][webtime-tracker] time tracker Chrome extension. The top half of sites, by average daily activity, for the last three months are as follows:

- `reddit.com`: 29 minutes and 25 seconds
- `chess.com`: 19 minutes and 51 seconds
- `youtube.com`: 11 minutes and 29 seconds
- `wikipedia.com`: 10 minutes and 44 seconds
- `facebook.com`: 6 minutes and 38 seconds
  - I use [this][newsfeed-eradicator] Chrome extension to hide my Facebook Newsfeed without blocking all of Facebook, which helps me control my usage here. I'm sure it was higher before doing this. The developers have recently extended it to cover Twitter, Hacker News and Reddit as well.
- `netflix.com`: 6 minutes and 19 seconds
  - I think this should probably be higher, because I usually watch Netflix while doing something else and I don't think the Chrome extension logs this time correctly

An important additional source is that I also have a local Tetris client called [Nullpomino][nullpomino] which I use intermittently, and the time from that doesn't get logged in the app. So all things told, I spend about a solid **hour to two hours a day** procrastinating. If I had to estimate the time without knowing these stats, I probably would have guessed less than 30 minutes.

## Results

I think I became a lot more cognizant of the strong pull of websites that encourage procrastinating after watching *The Social Dilemma*. It made me wish there was a service I could buy which would involve, like, someone standing over my shoulder and watching my screen and stopping me from doing work, although I would never actually sign up for a service that did this.

One thing that has been quite helpful has been meditation. My company reimburses a subscription for an app called [Headspace][headspace-app]. I tried this about a year ago, but didn't get much value out of it as I didn't find myself using it very often. Recently, though, I've started using it again and found it to be incredibly useful.[^headspace] I'm increasingly starting to think that data-driven[^headspace-experiment] meditation is a good alternative to data-driven procrastination.

[^crab-ganglion]: I'm reminded strongly of the lobster [stomatogastric ganglion][lobster-ganglion], which is a [model][central-pattern-generator] central pattern generator. For simplicity's sake, you can imagine two neurons connected to each other by inhibitory connections; when one turns on, the other turns off, until it tires itself out and the inhibition on the other weakens to the point where *it* can turn on. This creates a rhythmic activity which is useful for the lobster's digestive system. There are also neurons which produce bursting patterns independently simply through physical mechanisms (you can play with a model of this kind of neuronal behavior [here][neuron-models]).

[^procrastination]: When I was in elementary school, I would get up every morning before school to watch the previous night's episodes of *The Daily Show* and *The Colbert Report* while split-screening Chess or Tetris. As I got older and less willing to wake up early, this turned into using my computer in bed, mild insomnia, and experimenting with melatonin.

[^methods]: This actually involved a few steps. The first step was, since I have multiple computers and a phone, to move all of my "procrastination" time to one device and one screen. So I uninstalled Facebook from my phone, blocked the mobile web browser, and put my personal laptop in a cabinet. Since I use my computer a lot for legitimate work-related stuff, I split my work Chrome profile from my personal Chrome profile, and used uBlock to block a bunch of sites on my work profile, so that if I wanted to waste time I would have to open my personal profile.

[^headspace]: The usefulness for me personally is broken down into two buckets. First is the focus meditation series, which is a visualization exercise for how to cultivate more sustained, smoother focus sessions. It might be a placibo, but I feel noticably happier and more productive if I do a session in the morning. Second is the sleepcasts, which are fantastic. I habitually code before going to bed, and have a hard time turning my mind off. Instead of doing calming exercises to try to clear my mind, which I find hard to impossible, the sleepcasts redirect those thoughts into a more benign visualization which doesn't get in the way of sleeping. There are other features of the app which seem pretty useful but I haven't tried out yet. But for me these two components add significant value to my life and justify paying for the app.

[^headspace-experiment]: I suppose if I wanted to be truly data-driven about this as as well, it would be a good idea to A / B test this intervention by doing, for example, a week of meditation and a week without. Unfortunately I think I lack the scientific self-discipline to do this in any sort of rigorous manner.

[lobster-ganglion]: https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/stomatogastric-ganglion
[central-pattern-generator]: https://en.wikipedia.org/wiki/Central_pattern_generator
[neuron-models]: https://lightning.bolte.cc/#/posts/neuron_models
[webtime-tracker]: https://chrome.google.com/webstore/detail/webtime-tracker/ppaojnbmmaigjmlpjaldnkgnklhicppk?hl=en
[newsfeed-eradicator]: https://chrome.google.com/webstore/detail/news-feed-eradicator-for/fjcldmjmjhkklehbacihaiopjklihlgg?hl=en
[nullpomino]: https://github.com/nullpomino/nullpomino
[headspace-app]: https://www.headspace.com/headspace-meditation-app
