---
layout: post
title: "How Not to Productively Work From Home"
category: 🧐
excerpt: >
  Some of my thoughts about effectively controlling the elephant mind.
---

I'm lazy. Basically most of my time is spent doing meaningless stuff. I've watched every episode of "The Office" and periodically spend whole days playing League of Legends or Civ 5. I probably have a mild to medium internet addiction. I make a lot of goals, from exercising to eating healthier to sleeping at a more reasonable time, but I follow through on a very small number of them.

But people who know me generally find me pretty productive, and by most Capitalist metrics I've done pretty well for myself. If you're a subscriber to astrology, you'd chalk this up to my undirected Capricornian ambition, but as a Capricorn I don't believe in that sort of nonsense. So I thought it would be useful to get on my little soapbox and shout into the void about some things that I've been thinking about.

When I was in college, I went to a few meditation classes. I forget where I heard it originally, but I found the following idea to be really useful for thinking about productivity:

> You are an elephant trainer, and your mind is an elephant on a leash. You can't possibly control the elephant; you can only nudge on the leash and hope it goes that direction.

I'm not a biologist, but I have watched "The Lion King" (both the friendly, hand-drawn version and the Frankensteinian amalgum of CGI techniques), so I know a thing or two about elephants. Elephants aren't very smart. They don't look where they're going, and they often come very close to unapologetically trampling lion cubs. They just follow whatever path seems easiest, but they're extremely strong and can do a huge amount of work, like transporting and spraying water to celebrate the birth of a new loin heir.

If I have anything resembling a "productivity philosophy", it would be this: most of the time I'm not very cognizant of what I'm doing (when my brain is being controlled by elephant mind), so in the few moments that I am, I should try to lay out a nice path for the elephant. The tricky part is figuring out how to do that well.

You can't force the elephant mind to do anything. If you adopt an attitude of trying to force the elephant mind into submission, the elephant mind will just wait around until you start to get bored, and tip-toe around your back. This has been my experience using browser extensions like Blocksite and screen time tools for Apple and Android products. It is the technological Achilles heal of both productivity hackers and youth group leaders trying to guilt young men into not masturbating. Indeed, I suspect that the elephant mind of any sufficiently productive software engineer will be able to disable pretty much any productivity software that you tries to give them.

As an illustration of this, if you spend some time Googling "how to block websites as a super smart unix programmer i'm serious", you'll find something saying that you can modify the `/etc/hosts` file to make websites redirect to localhost, effectively blocking them. This is how that normally plays out for me.

````bash
$ echo "127.0.0.1 reddit.com" >> /etc/hosts
$ # Enjoy a solid few hours of productivity
$ # Friend sends me a meme they found on Reddit
$ # Try to access it and remember that I disabled Reddit in my hosts file
$ # ...
$ vim /etc/hosts  # Sheepishly
```

If I'm feeling particularly persistent, it'll go something like this:

```bash
$ echo "127.0.0.1 reddit.com" >> /etc/hosts
$ chmod -w /etc/hosts
$ # ...
$ chmod +w /etc/hosts
$ vim /etc/hosts  # Sheepishly
```

Alternatively, at one point I realized I could block websites through my router. Actually, the TPLink routers have this whole setup that lets you limit the amount of time you spend on the internet everyday and which sites you visit. Unfortunately, since I go on the internet professionally, most of time the first option doesn't work for me; and regarding the second, my elephant mind knows about VPNs.

**This isn't to say** that these kinds of tools and techniques aren't useful. **It is to say** that you won't be able to depend on them to do all of the work required to get your elephant mind to focus on important stuff.

Here are some more things I've found to be losing strategies:

- Guilting yourself into being productive. I've found that, while this might work short-term, at some point the elephant mind just stops caring about what you think.
- Making work seem more important than it is. While some stuff can be painted this way - who hasn't tried to write an essay in the dying hours before it's due - it's not a great strategy for long-term productivity. Besides that, you end up stressing yourself out. Not great.
- Trying to make thing seem easier than they are. I thought at one point that if I just dedicated itself to coming up with easier ways to do different kinds of tasks, then my elephant mind would find them more enjoyable and be more predisposed to do them. Unfortunately, some things just can't be made easier. In fact, I would argue that the more difficult something is, the more worthwhile it tends to be.

So the question is, how do you go about convincing the elephant mind to do your bidding? I like to think about how I'd convince a little kid to do something I want. The best thing seems to be to try to really convince yourself about the general importance of hard work. Reward your elephant mind whenever you see it doing something difficult by giving it a mental pat on the back.
````
