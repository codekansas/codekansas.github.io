---
layout: post
title: "Possible Ideas for Robotics Startups"
category: 🧐
excerpt: >
  A list of ideas for robotics startups.
---

Here are some ideas for robotics startups.

* TOC
{:toc}

## Present-Day Ideas

This section is for ideas which I think are feasible to build using current robotics techniques (building maps, using explicit poses, grasping - the "NVIDIA" robotics approach, if you will).

### Grocery

- Collect shopping carts from parking lots
  - There are robotic systems which help push long lines of shopping carts, so that one person can go around and collect shopping carts from a parking lot. This could likely be automated using conventional robotics approaches.
- Move pallets automatically (based on a map)
  - In warehouses or bulk suppliers, there's usually a bunch of pallets that need to be moved around. This means loading and unloading trucks, keeping track of where things are, etc. There are fork lifts and other machines for doing this, but parts of it could likely be automated.
- Restocking shelves
  - There's a pretty clear service / subscription business model for a robot which can automatically restock shelves
  - It is possible to build a robotic restocking system using current machine learning techniques: localize the robot in the environment, build a static map, match items, keep track of inventory
  - Can expand into an "Amazon Go" style system by combining inventory software

### Supply Chains

- Assembly line picking
  - Essentially, describe some item that you want to pick out of a stream of items, and have a robot arm or suction cup pick out all instances of that item.
  - This is a semi-novel research problem in that it involves using natural language to specify the items in question, instead of building an explicit classifier, but the picking system would likely be done using some conventional robotics approach
- Automatic food delivery
  - There are a lot of companies trying to enter this space (with dubious success)
  - This would likely be ideal for New York, although maybe there is a natural resistence (try riding a bike around New York)

### Waste Management

- Remove recyclable materials from waste stream
  - This is what Amp Robotics is doing
  - Many recycling plants have developed highly sophisticated machinery for doing recycling at scale:
    - Using different bands of light to identify types of plastic
    - Air blowers to quickly separate out plastic shards
- Collecting waste from homes and businesses (carting)
  - Similar to automatic food delivery, but the reverse side
  - Aluminum can recycling:
    - It would be really cool to melt down aluminum cans somewhere nearby New York, so that they can be sold off more conveniently (instead of shipping a bunch of compressed cans which need to be melted down somewhere else).
    - Aluminum smelting produces waste products, probably needs to be done in specific zones because of air quality concerns
  - Plastic recycling:
    - Being able to 3D print things nearby New York would reduce the costs associated with delivery and collection (although maybe this is not a large cost)

### Medical

- Automated gurney
  - Some patients are very heavy
  - A gurney which would make it easier to transport patients (or could even take the patient to a specific location entirely automatically) would remove the need for having a human pushing them around
- Automatic blood collection
  - This is a pretty interesting problem, because it involves a lot of different types of sensors and actuators, and a lot of different types of data (e.g. blood pressure, blood oxygen, etc.)
  - It would be really cool to have a system which could automatically collect blood samples from patients, and then send them to a lab for analysis
  - This would be a pretty good use case for a robot which can do a lot of different things, and would be able to do a lot of different things in a hospital environment
- Robotic blood analysis
  - This is an active space, but may not be the best application for a machine learning startup to try to tackle
- Robotic surgery
  - Systems like Davinci are already pretty good at this, but there is still a lot of room for improvement
  - Automating parts of the surgery is an extra challenging problem for machine learning systems, because it involves a lot of uncertainty about the environment and the task
- Robot nurse
  - Replacing aspects of the nurses' job, such as:
    - Taking vitals
    - Moving patients
    - Delivering food
    - Delivering medicine
    - Cleaning

### Home

- Robot vacuum
- Assistance with moving boxes and furniture
  - This actually seems pretty difficult because of the range of environments and things that someone would want to move
  - A robotic system could potentially help carry most things, leaving a few more tricky things to be carried by hand. For example, it could be really good at carrying a specific size of box
- Robotic lawn mower
  - There are a few companies working in this space already
  - Specific machine learning based improvements over existing methods include:
    - Using a camera to detect obstacles
    - Optimizing the path to minimize the amount of time spent mowing
- Window washing
  - One of the most common jobs in New York is window washing
  - Useful for tall buildings
- Robotic dog walker
- Robot babysitter
  - Similar to the pet monitoring robots, but for children

### Old Age

- Mobility robot
  - An old person who wants to get from one place to another would ask the robot to take them to a specific location
- Robot companion
  - A robot which can talk to an old person, and help them with their daily tasks, such as:
    - Taking them to the bathroom
    - Using a camera to detect if they are eating properly
    - Assisting them with their medication
    - Providing a teleop platform for their family to interact with them

Some other problems that the elederly face which might be solvable using robotics:

- Falls
- Dementia
- Alzheimer's
- Parkinson's
- Depression
- Loneliness
- Isolation
- Incontinence
- Malnutrition
- Medication errors
- Social isolation

## Future Ideas

Given the current state of machine learning, the big upcoming wave of robotics ideas will likely be systems which can understand very open-ended environments and tasks, and deal with large amounts of ambiguity. Consider a system like ChatGPT, but trained using representations of the physical environment in addition to large-scale internet text.

- The form factor for the system would have to be somewhat large, because of the computational requirements.
- It would be well-suited to solving tasks which don't have good solutions using current robotics systems.
- At scale, there is a benefit to building general-purpose robots, but this benefit likely will be realized by a large company with the resources to produce at scale rather than a small startup.
  - The caveat to this is that the benefits for the large company would almost entirely be in efficient hardware manufacturing rather than on the software side. It is essentially free to collect unlabeled data about the world.
  - Is there any intrinsic reason for economies of scale with regards to manufacturing robots? The right robotic system might not actually have an economy of scale.

So, to summarize the essential requirements for a startup to compete in five to ten years:

1. Low hardware cost
2. Large form factor
3. General purpose
  - General mobility: Multiple legs
  - General manipulation: Multiple arms

### Quadroped

A Spot-like quadroped robot, with an arm attached, could be relatively cheap to manufacture and would be able to do a lot of useful tasks, such as:

- Carrying things from one place to another
- Monitoring an environment
- Using tools

Additionally, the extra stability provided by four legs would allow it to carry a larger battery and onboard computer, enabling it to do more sophisticated world understanding.

### Biped

Tesla's robot looks to be a pretty cool platform for interacting with the world. The question is, could a startup manufacture a bipedal robot which would be a better value proposition than Tesla's? I think most people would probably say no, but here are some arguments in favor of the startup:

- I don't think Tesla's machine learning stack is very good compared to alternatives like Nvidia, Tenstorrent, Graphcore, etc. There are so many edge acceleration startups out there that are, in my opinion, doing a better job than Tesla - or more precisely, the fact that they have to build their stack to serve external customers *forces* them to build a better stack. My experience was that at Tesla, the fact that the only consumer was internal meant that the team working on the compiler stack was able to get away with a lot of extremely hacky stuff.
- On the hardware front, building a robot is comparatively simple compared to building a car. There are fewer regulatory hurdles at present, and the bill of materials is a lot cheaper, especially using commodity hardware.
