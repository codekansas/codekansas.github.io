---
layout: post
title: Humanoid Robot Tracking Post
tags: [robotics]
excerpt: >
  A tracking post with all of my notes on making a humanoid robot.
---

## References

- [Shadow Robot Hand][video-shadow-hand]
- [Fourier Intelligence GR-1][video-fourier-gr1]

## Hand

I recently got into a bike accident and was worried that I might have fractured my hand (according to my wife, the way that I fell and the injury location was in line with the classic board exam description of a [Colles' fracture][colles-fracture]). Fortunately, the X-ray came back negative, but it was the first time I've actually gotten to see my own hand bones. My wife has also mentioned to me that hand anatomy is one of the most complicated things to memorize in medical school, which I found interesting having worked in robotics because grasping robots tend to be quite simple.

To rectify this apparent disparity, I figured it would be a good opportunity compare my hand X-rays with a few robot hands I found online. Hopefully this will be useful to roboticists working on grasping and manipulation.

First, some medical terminology:

1. In medicine, people use _dorsal_ and _ventral_ to refer to where things are on the body. With regards to hands, _dorsal_ is the back of the hand and _ventral_ is the palm.
2. Similarly, _abductors_ and _adductors_ refer to mucles that move things away from and towards the body, respectively. For example, the _abductor pollicis brevis_ is the muscle that moves the thumb away from the palm, while the _adductor pollicis_ is the muscle that moves the thumb towards the palm.
3. This means that _abductor_ muscles tend to be on the _dorsal_ side of the hand, while _adductor_ muscles tend to be on the _ventral_ side of the hand.

### Human Hand

Here's a copy of my hand X-ray, showing the bones in the hand and wrist.

{% include figure.html image="/images/hand-xray/full.webp" caption="The raw set of X-rays from my recent hospital visit. The top rows are the three hand X-rays, while the bottom rows are the three wrist X-rays." %}

Here's individual links to each of the X-rays as well:

- <a href='/images/hand-xray/hand-1.webp' target='_blank'>Hand 1</a>
- <a href='/images/hand-xray/hand-2.webp' target='_blank'>Hand 2</a>
- <a href='/images/hand-xray/hand-3.webp' target='_blank'>Hand 3</a>
- <a href='/images/hand-xray/wrist-1.webp' target='_blank'>Wrist 1</a>
- <a href='/images/hand-xray/wrist-2.webp' target='_blank'>Wrist 2</a>
- <a href='/images/hand-xray/wrist-3.webp' target='_blank'>Wrist 3</a>

Here's an image of the dorsal hand and forearm muscles, from [here][muscles-dorsal].

{% include figure.html image="/images/hand-xray/dorsal-muscles.webp" caption="The muscles on the dorsal side of the hand and forearm (meaning, the back of the hand)." %}

Here's an image of the ventral hand and forearm muscles, from [here][muscles-ventral].

{% include figure.html image="/images/hand-xray/ventral-muscles.webp" caption="The muscles on the ventral side of the hand and forearm (meaning, the palm of the hand)." %}

### Robot Hand

Here's a very simple robot hand that I found on Amazon [here][amazon-robot-hand].

{% include figure.html image="/images/hand-xray/amazon-hand.webp" caption="A simple robot hand that I found on Amazon." %}

Here's the more sophisticated Shadow Hand from [Shadow Robot Company][shadow-hand].

{% include figure.html image="/images/hand-xray/shadow-hand.webp" caption="A more complex robot hand, featuring 17 degrees of freedom. Each DOF needs two actuators for abduction and adduction." %}

### My Design

I'm in the process of designing my own version of a robot hand, roughly modeled on a real hand. To start with, here is the design of a finger:

{% include figure.html image="/images/hand-xray/cad-finger.webp" caption="A single finger designed in CAD, with holes for placing zip ties and threading to represent tendons." %}

Some key features of this design:

1. The knuckles don't allow the finger to rotate backwards

[amazon-robot-hand]: https://www.amazon.com/Fingers-Movement-Bionic-Mechanical-DIY%EF%BC%88Left/dp/B081RRCTFX/
[colles-fracture]: https://my.clevelandclinic.org/health/diseases/21860-colles-fracture
[muscles-dorsal]: https://www.osmifw.com/hand-therapy-center-in-fort-worth/hand-and-wrist-injuries-and-disorders/forearm-muscles-dorsal-compartment-2/
[muscles-ventral]: https://boneandspine.com/forearm-muscles/
[shadow-hand]: https://www.shadowrobot.com/dexterous-hand-series/
[video-shadow-hand]: https://www.youtube.com/watch?v=3ju4upwhdvM
[video-fourier-gr1]: https://www.youtube.com/watch?v=KoAEaZm1Hw4
