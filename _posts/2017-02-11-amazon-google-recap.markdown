---
layout: post
title: "What It's Like to Intern at Amazon and Google"
date: 2017-02-11 12:00:00
categories: personal
keywords:
 - Amazon
 - Google
 - Internship
image: /resources/index/gomazon.png
excerpt: >
  An overview of my internship experiences at Amazon and Google (I will probably update this after I finish at Facebook).
---

All the opinions I express in this post are strictly my own.

**TL;DNR** Amazon is a good place to go if you want to do things well and be pushed to do them better. Google is a good place to go if you want to be around the smartest people in the world and spend a lot of time being creative.

# Overview

I interned at Amazon in Seattle in the summer of 2016, and Google in Mountain View in the fall. The teams I worked on were pretty different, so the comparison isn't necessarily fair, but hopefully I'll provide some useful advice for people looking at interning at big tech companies.

<center><h1>Amazon</h1></center>

# My project

I worked on the Amazon India team. Over the course of the internship, I built several APIs to integrate with India's invoice management service, which my team owns. I also built a web service for interacting with each of the APIs. My project was closely tied to another intern's project, so we both got very familiar with using Git to work simultaneously on the same code base. Additionally, near the end of the internship I was pretty much done with my project, and I got to diagnose and solve a Sev 2 issue.

My knee-jerk reaction to building APIs and a website is, "this seems pretty plain vanilla, I wouldn't want to do that". However, there was a lot of nuance and complexity that made it challenging and engaging, and once I was done I could clearly see the impact it would have. I was very pleasantly surprised with how much I enjoyed working on it and the level of ownership I felt. I was also really impressed that they would give an intern the responsibility to handle a high severity issue.

# Crying at work

Based on [a certain New York Times article](http://www.nytimes.com/2015/08/16/technology/inside-amazon-wrestling-big-ideas-in-a-bruising-workplace.html), I thought Amazon was going to be high-stress, high-demand on a constant basis. That definitely wasn't my experience. Most people I asked about the article also said it didn't reflect their experience. However, it isn't taken for granted that you're good at your job. As another SDE said, Amazon tends to end up with people who have high expectations already, so while I never saw anyone cry at work, I would imagine it would be more from internal pressure than external. It felt what I'd imagine a startup feels like.

Additionally, I should say how awesome my team was. Everyone was extremely competent. There was ample support if I got stuck. I never felt overworked or underworked. Ping Pong was very popular in our office.

# Interview process

I don't want to (and probably can't) talk about specific interview questions, but I will give some of my personal background and prep work. I spent a lot of time making Flash games in middle school, but did practically no coding throughout high school, and only started in force once I got to college. When I started applying, I had taken about two and a half years of computer science classes. Other people have written about what makes people good or bad at computer science, but in my opinion, the most important things are to be okay with failing and to be willing to put in enough effort. A strong math background also doesn't hurt (many interviews I did seemed to test my ability to think mathematically). And you should be able to sit and work on one thing for about an hour at a time without getting distracted.

I didn't prepare much for interviews, but I did study for the ACM and I like to compete in online coding competitions (the Google Code Jam and Facebook Hacker's Cup are good practice, besides being things you can put on your resume). An algorithms class is usually necessary. If you have never looked at [Cracking the Coding Interview](https://www.amazon.com/Cracking-Coding-Interview-Programming-Questions/dp/098478280X) I would really, really recommend getting a copy, or getting with a friend who has a copy. You should be able to figure out the best answers to the Hard problems with some assistance. There are some questions which have interesting tricks.

Lastly, even if you don't go to a big Tech school, it is totally possible to land a decent internship, although the lack of name recognition may hurt. I go to Emory University in Atlanta, and our computer science program is dwarfed by Georgia Tech. If you're in a similar position, I found that reaching out to people directly really helped. Use your school's Alumni network and send personal emails to recruiters.

<!--
    __    _ ____        __  __           __
   / /   (_) __/__     / / / /___ ______/ /__
  / /   / / /_/ _ \   / /_/ / __ `/ ___/ //_/
 / /___/ / __/  __/  / __  / /_/ / /__/ ,<   
/_____/_/_/  \___/  /_/ /_/\__,_/\___/_/|_|  

If, like my case, your school isn't frequented by tech recruiters, here is a life hack: Put on LinkedIn that you're a computer science major graduating this year, and you will probably get a lot more emails from recruiters who are just spamming LinkedIn's search tools. Then, when they email you about applying for a job, ask them to put you in contact with an intern hiring manager. Just kidding. Maybe.
-->

Amazon's interviews were speedy, probably the shortest start-to-offer time of any company I applied for. You can probably check Glassdoor for non-anecdotal evidence, but most people I asked had similar experiences. For me it was about two weeks. I didn't think the interview went super well, but I solved the problem they presented me. Some other interns I talked to didn't even have coding interviews.

# Non-compete agreement

An annoying thing about Amazon (and many tech companies, although I've heard it's more strict at Amazon) is that you have to [sign an NDA and non-compete agreement](https://www.quora.com/Why-does-Amazon-require-an-NDA-in-order-to-interview-with-their-company) while working there. Part of the contract means that if you want to work on outside projects of any kind (for me, contributing to open-source projects on Github or working on my Master's thesis) you have to get it cleared with Human Resources. This is so that you don't accidentally work on something that another part of Amazon is working on. This might be standard practice for many companies, but it felt weird when I started.

# Internal projects

Amazon has some [really cool internal projects](http://www.techinsider.io/amazon-secret-projects-2015-8), but there is a decent amount of secrecy even within the company. But for people concerned that Amazon isn't leading innovation in certain areas, they are certainly bulking up their research. I got to go to the [DSSTNE](https://github.com/amznlabs/amazon-dsstne) release talk, which was really cool. Overall, the vibe I got from the Personalization team (which does a lot of the Machine Learning work) was very pragmatic. For example, DSSTNE was made to emphasize support for only vanilla neural network architectures, but with sparse matrices (all of this is public knowledge; you can infer what this might mean for how they do product recommendations).

<center><h1>Google</h1></center>

# My Project

I worked on the handwriting recognition team at Google, better known as the team responsible for [Quick, Draw!](https://quickdraw.withgoogle.com/) This was a very different type of team from the one I was on at Amazon; most of the people I was around had advanced degrees, and the team was research-oriented (it was within the Machine Perception research group). I worked on improving the mixed-script handwriting recognition, as well as some random other projects.

This internship was a huge learning experience. At Amazon, everything seemed pretty well structured, and there was a pretty clear end goal which I could work towards. At Google, the project was much more free-flowing, which was a little awkward for me to get use to, but also meant I had the opportunity to learn a ton.

# Amazing People

The main thing that struck me about being at Google was the sheer number of amazing people I'd see around the campus. It feels like the people there are working on building the future. I was playing around with some stuff with Generative Adversarial Networks, and got to see Ian Goodfellow, the guy who invented them, give an early version of his NIPS talk. It was pretty amazing and inspiring to be around such incredible people, and it gave me the feeling that I could do anything I wanted to.

Related to this, I had a much stronger feeling like this was a really special opportunity, and that I should work really hard to make it worthwhile. I spent a lot more time at Google than I did at Amazon, although it definitely wasn't expected. I noticed that Googlers tended to work longer hours in general; it wasn't uncommon to see people around the campus on weekends and later at night.

# Perks

The Google perks were nice for the first few weeks, but I felt like I just got used to them after a time. The free food was the biggest one; it meant I didn't have to think about getting food, which freed up some of my brainspace. The campus was pretty awesome, although I didn't find myself spending much time using random stuff like the rock wall. I did like their soccer fields, which usually had people playing pick-up.

# Internal Tools

Google's internal tools make Amazon's seem like they're from the stone age. Their build system, `blaze`, is pretty amazing (there is an open-source version called [Bazel](https://bazel.build/)). Plus, since most of the stuff I was doing used TensorFlow, it felt really, really natural to write code and put it up on their big Borg cluster.

# Interview Process

This was my least favorite part about Google; their interview process is really complicated. When I applied for a Summer internship, I went through three phone interviews, then got put in the "host matching" pool. I sat around in it for about two months, during which time I interviewed for and got an offer from Amazon, then found out that I didn't get any follow-up interviews. So basically you can only get an internship if someone at Google sees your resume and likes you.

I think this might be a drawback for people from less well-known schools, since it isn't as data-based. When I got to Google in the fall, I got to look at the other side of this process. Without breaking any NDAs, it's pretty much a webpage that people look through. Some people are "starred" but most people aren't. You can filter by related interests. My recruiter said I would have the best chances if I made it seem like I was open to doing anything, but when I did that in the summer, no one was interested in me. I think better advice is to taylor your resume towards a specific area, and say you're interested in that area, because when hosts are matching people they are looking for people who are interested in their area. Plus, if you do get an offer, you'll end up doing exactly what you want to do.
