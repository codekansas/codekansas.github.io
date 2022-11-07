---
layout: post
title: "Getting Started with ROS 2"
category: 🔬
excerpt: "An actually good guide to getting started with ROS 2"
---

In this post, I'm going to provide some straight-forward pointers for getting started using ROS 2.

* TOC
{:toc}

## Introduction

I recently started a new job which involves me working on the [Stretch][stretch-robot] robot from Hello Robots. Since my background is principally in machine learning and not robotics, this is my first time using ROS. However, when I searched for "ROS 2 API" or "How to use ROS 2" or "Getting Started with ROS 2" most of the sites I found looked like they had been insanely SEO'd to the point of being unusable. To wit, I figured I would document my progress as I am learning so that others who may be making the same transition might benefit as well.

### What this Post Covers

In this blog post I hope to do the following:

1. Provide a simple on-boarding guide building a system in ROS 2 to control a robot
  - Specifically, I'll be using the [JetBot][jetbot], since that's what I have on hand, but since I'm also going to be using the Stretch at work this guide will hopefully prove useful if you're working on manipulation as well
  - I'm going to be somewhat opinionated about the "right" way to do things (mostly based on my background in building other types of software) so I may leave out some alternative approaches, but I'll try to be very thorough in describing how to do things
2. Make a searchable reference for how to do simple things
3. Provide some more in-depth technical explanations for how different ROS components work, and why they work that way
  - This is more because I think it's important to understand the inner workings of things in order to debug and optimize them, but I'll try to structure the post in such a way that these parts are easy to skip over if you just want to get started building something.

If I miss anything, please leave a comment (note that the commenting tool requires a Github account).

### What this Post Doesn't Cover

In this post I won't be covering:

- The original version of ROS (it's on the way towards no longer being supported)
- Python stuff (I'm assuming a basic level of familiarity with Python, Anaconda, version control, and other stuff like that)

## Getting Started

Throughout this post, I will be using [RoboStack][robostack], which is a way of installing ROS as a Conda bundle. This makes it easier to manage multiple versions of ROS and install new packages. I'll assume you have Miniconda installed somewhere on your system - if not, install it from [here][miniconda-install]. Then, to get started, run:

{% highlight bash %}
# Robostack's ROS 2 Humble channel only works with Python 3.9.
# Galactic can work with either version.
conda create --name ros-blog-post python=3.9
conda activate ros-blog-post
{% endhighlight %}

Next, install [Mamba][mamba]:

{% highlight bash %}
conda install -c conda-forge mamba
{% endhighlight %}

Mamba is a drop-in replacement for the Conda CLI which is a lot faster and makes working with Conda packages a lot easier.

> If you want to avoid adding specific Conda channels, such as `-c conda-forge`, you can add them to your `~/.condarc` file using `conda config --add channels conda-forge`. Then you can just do something like `conda install mamba` and it will look in the `conda-forge` channel automatically. In particular for this post I suggest adding the `robostack-humble` channel.

Next, install the Humble distro of ROS 2 (from the installation instructions [here][ros-2-humble-install-instructions]):

{% highlight bash %}
# Install Humble distro.
mamba install \
  -c robostack-humble \
  -c conda-forge \
  spdlog=1.9.2 \
  foonathan-memory=0.7.2 \
  ros-humble-desktop

# These are the instructions to install the Galactic
# distro, which worked fine for me on an Ubuntu machine
# but failed on my M1 Mac.
mamba install \
  -c robostack-experimental \
  ros-galactic-desktop
{% endhighlight %}

Note that installing ROS 2 this way adds some scripts to the directory in `${CONDA_PREFIX}/etc/conda/activate.d/`. These set some environment variables which are important. In order to get these scripts to run, you have to restart your Conda environment, like so:

{% highlight bash %}
conda deactivate
conda activate ros-blog-post
{% endhighlight %}

You can double-check that the installation was successful by running the dummy programs. In one terminal session, run:

{% highlight bash %}
ros2 run demo_nodes_cpp talker
{% endhighlight %}

If everything was installed correctly, you should see something like this:

{% highlight text %}
[INFO] [1667791882.591136612] [talker]: Publishing: 'Hello World: 1'
[INFO] [1667791883.591488073] [talker]: Publishing: 'Hello World: 2'
[INFO] [1667791884.595503927] [talker]: Publishing: 'Hello World: 3'
[INFO] [1667791885.591661410] [talker]: Publishing: 'Hello World: 4'
[INFO] [1667791886.593317378] [talker]: Publishing: 'Hello World: 5'
{% endhighlight %}

In another terminal session, run:

{% highlight bash %}
ros2 run demo_nodes_cpp listener
{% endhighlight %}

If everything worked as expected, you should see something like this:

{% highlight text %}
[INFO] [1667791947.122910896] [listener]: I heard: [Hello World: 13]
[INFO] [1667791948.122430228] [listener]: I heard: [Hello World: 14]
[INFO] [1667791949.123189150] [listener]: I heard: [Hello World: 15]
[INFO] [1667791950.119661010] [listener]: I heard: [Hello World: 16]
[INFO] [1667791951.121229768] [listener]: I heard: [Hello World: 17]
{% endhighlight %}

## ROS 2 Commands

### `ros2 topic list`

List all topics:

{% highlight bash %}
ros2 topic list -t
{% endhighlight %}

### `ros2 service list`

List all services:

{% highlight bash %}
ros2 service list
{% endhighlight %}

[miniconda-install]: https://docs.conda.io/en/latest/miniconda.html
[stretch-robot]: https://hello-robot.com/stretch-2
[jetbot]: https://jetbot.org/master/
[robostack]: https://robostack.github.io/
[mamba]: https://anaconda.org/conda-forge/mamba
[ros-2-humble-install-instructions]: https://github.com/RoboStack/ros-humble
