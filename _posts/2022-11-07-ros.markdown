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

Throughout this post, I will be using [RoboStack][robostack], which is a way of installing ROS as a Conda bundle. This makes it easier to manage multiple versions of ROS and install new packages.

- I'll assume you have Miniconda installed somewhere on your system - if not, install it from [here][miniconda-install].
- I'm running this on my Mac M1 machine, which is monotonically worse for ROS than using Ubuntu, meaning that any command that works in this tutorial will work with an Ubuntu machine, but not the other way around.

### Create a New Conda Environment

{% highlight bash %}
# Robostack's ROS 2 Humble channel only works with Python 3.9.
# Galactic can work with either version.
conda create --name ros-blog-post python=3.9
conda activate ros-blog-post
{% endhighlight %}

### Install [Mamba][mamba]

{% highlight bash %}
conda install -c conda-forge mamba
{% endhighlight %}

Mamba is a drop-in replacement for the Conda CLI which is a lot faster and makes working with Conda packages a lot easier.

> If you want to avoid adding specific Conda channels, such as `-c conda-forge`, you can add them to your `~/.condarc` file using `conda config --add channels conda-forge`. Then you can just do something like `conda install mamba` and it will look in the `conda-forge` channel automatically. In particular for this post I suggest adding the `robostack-humble` channel if you plan on using ROS 2 Humble.

### Install the Humble distro of ROS 2

Following the installation instructions [here][ros-2-humble-install-instructions]:

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

### Install Colcon

ROS 2 packages are built using [colcon][colcon-info], so you will need to install it as well:

{% highlight bash %}
mamba install \
  -c conda-forge \
  colcon-core \
  colcon-common-extensions
{% endhighlight %}

### Check that Installation Succeeeded

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

## Turtlesim

The introductory ROS node is [Turtlesim][turtlesim-ros-page]. To get started, simply run

{% highlight bash %}
ros2 run turtlesim turtlesim_node
{% endhighlight %}

If you are running this on your local machine, it should pop up a window that looks like this:

![](/images/ros/turtlesim.png)

The command above starts a new [node](#node) which manages the window. In another terminal, start another [node](#node) which takes keyboard inputs and communicates with the associated [topics](#topic) and [actions](#actions) on the first node:

{% highlight bash %}
ros2 run turtlesim turtle_teleop_key
{% endhighlight %}

You can also directly call a topic by using the command below:

{% highlight bash %}
ros2 topic pub /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0}, angular: {z: -3.1415926535}}"
{% endhighlight %}

If everything is working as expected, this should make the turtle spin around in circles on your screen.

### Writing a Custom Node

Let's write a new node to interact with the Turtlesim node. First, let's create a new workspace somewhere:

{% highlight bash %}
mkdir ros-blog-post
cd ros-blog-post
git init .
{% endhighlight %}

Next, create a new package using the command:

{% highlight bash %}
ros2 pkg create --build-type ament_python custom_turtlesim
{% endhighlight %}

If you run the `tree` command (if it's not installed, see [these instructions][tree-install-instructions]), you should get a directory structure that looks like this:

{% highlight bash %}
$ tree
.
└── custom_turtlesim
    ├── custom_turtlesim
    │   └── __init__.py
    ├── package.xml
    ├── resource
    │   └── custom_turtlesim
    ├── setup.cfg
    ├── setup.py
    └── test
        ├── test_copyright.py
        ├── test_flake8.py
        └── test_pep257.py
{% endhighlight %}

Next, create a new file in `custom_turtlesim/custom_turtlesim/controller.py` and add the following:

{% highlight python %}
import math
from typing import List, Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class TurtlesimController(Node):
    def __init__(self, run_every_n_seconds: float = 0.5, queue_size: int = 10) -> None:
        """Defines a simple controller for Turtlesim.

        Args:
            run_every_n_seconds: Run the publisher every N seconds
            queue_size: The number of messages to queue up if the subscriber
                is not receiving them fast enough; this is a quality-of-service
                setting in ROS
        """

        super().__init__(node_name="turtle_sim_controller")

        self.turtle_pub = self.create_publisher(Twist, "/turtle1/cmd_vel", qos_profile=queue_size)
        self.timer = self.create_timer(run_every_n_seconds, self.timer_callback)

    def timer_callback(self) -> None:
        """Defines a callback which is called every time the timer spins."""

        msg = Twist()
        msg.linear.x = 1.0
        msg.angular.z = math.pi
        self.turtle_pub.publish(msg)
        self.get_logger().info("Published a message")


def run_turtlesim_controller(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)

    controller = TurtlesimController()

    try:
        rclpy.spin(controller)
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    run_turtlesim_controller()
{% endhighlight %}

### Running the Custom Node

#### Ad-hoc

There are two ways to run this. The simplest is to simply run the Python script directly:

{% highlight bash %}
# Run the script.
python custom_turtlesim/custom_turtlesim/controller.py

# Run as a module.
cd custom_turtlesim/
python -m custom_turtlesim.controller
{% endhighlight %}

This way you can do quick debugging without having to rebuild your entire package

#### Full Build

When you're done debugging your controller, you can build the whole package and add it to your environment.

Add the `run_turtlesim_controller` function as an entry point in the file `custom_turtlesim/setup.py`:

{% highlight bash %}
entry_points={
    "console_scripts": [
        "controller = custom_turtlesim.controller:run_turtlesim_controller",
    ],
},
{% endhighlight %}

From your root `ros-blog-post/` directory, build the package by running:

{% highlight bash %}
colcon build
{% endhighlight %}

Now your directory tree should look something like this:

{% highlight bash %}
$ tree -L 1
.
├── build
├── custom_turtlesim
├── install
└── log
{% endhighlight %}

Install the newly-built package:

{% highlight bash %}
. install/setup.bash  # If using bash
. install/setup.zsh   # If using zsh
{% endhighlight %}

Finally, you can run your service using:

{% highlight bash %}
ros2 run custom_turtlesim controller
{% endhighlight %}

You can run the auto-generated tests using

{% highlight bash %}
colcon test
{% endhighlight %}

## Terminology

### Package

A *package* in ROS is a collection of code, much like a Python or C++ package.

To create a new Python package:

{% highlight bash %}
ros2 pkg create --build-type ament_python <package-name>
{% endhighlight %}

To create a new C++ package:

{% highlight bash %}
ros2 pkg create --build-type ament_cmake <package-name>
{% endhighlight %}

To prepopulate with a starter [node](#node), you can add the command line argument `--node-name <node-name>`

### `package.xml`

All ROS packages include a `package.xml` file which contains meta information about the package. However, when using RoboStack as your dependency manager, you can more or less ignore it.

### Node

A *node* in ROS is a process that performs some computation - basically an executable. They communicate with each other using [topics](#topic), [services](#service) and [actions](#actions). You can list *running* nodes using

{% highlight bash %}
ros2 node list
{% endhighlight %}

You can print out info about a particular running node using

{% highlight bash %}
ros2 node info <node-name>
{% endhighlight %}

<details>
<summary>Sample output when running <code>turtlesim_node</code></summary>
<div>
{% highlight bash %}
$ ros2 node info /turtlesim
/turtlesim
  Subscribers:
    /parameter_events: rcl_interfaces/msg/ParameterEvent
    /turtle1/cmd_vel: geometry_msgs/msg/Twist
  Publishers:
    /parameter_events: rcl_interfaces/msg/ParameterEvent
    /rosout: rcl_interfaces/msg/Log
    /turtle1/color_sensor: turtlesim/msg/Color
    /turtle1/pose: turtlesim/msg/Pose
  Service Servers:
    /clear: std_srvs/srv/Empty
    /kill: turtlesim/srv/Kill
    /reset: std_srvs/srv/Empty
    /spawn: turtlesim/srv/Spawn
    /turtle1/set_pen: turtlesim/srv/SetPen
    /turtle1/teleport_absolute: turtlesim/srv/TeleportAbsolute
    /turtle1/teleport_relative: turtlesim/srv/TeleportRelative
    /turtlesim/describe_parameters: rcl_interfaces/srv/DescribeParameters
    /turtlesim/get_parameter_types: rcl_interfaces/srv/GetParameterTypes
    /turtlesim/get_parameters: rcl_interfaces/srv/GetParameters
    /turtlesim/list_parameters: rcl_interfaces/srv/ListParameters
    /turtlesim/set_parameters: rcl_interfaces/srv/SetParameters
    /turtlesim/set_parameters_atomically: rcl_interfaces/srv/SetParametersAtomically
  Service Clients:

  Action Servers:
    /turtle1/rotate_absolute: turtlesim/action/RotateAbsolute
  Action Clients:
{% endhighlight %}
</div>
</details>

### Topic

A *topic* in ROS is a way for [nodes](#node) to send messages to each other using publish/subscribe semantics. A node will publish a topic, which can then be subscribed to by other nodes. You can list available topics using:

{% highlight bash %}
ros2 topic list
{% endhighlight %}

To show the topic type, you can use:

{% highlight bash %}
ros2 topic list --show-types
ros2 topic list -t  # Shorthand
{% endhighlight %}

<details>
<summary>Sample output when running <code>turtlesim_node</code></summary>
<div>
{% highlight bash %}
$ ros2 topic list --show-types
/parameter_events [rcl_interfaces/msg/ParameterEvent]
/rosout [rcl_interfaces/msg/Log]
/turtle1/cmd_vel [geometry_msgs/msg/Twist]
/turtle1/color_sensor [turtlesim/msg/Color]
/turtle1/pose [turtlesim/msg/Pose]
{% endhighlight %}
</div>
</details>

To send a message to a topic from the command line, you can use the command:

{% highlight bash %}
ros2 topic pub <topic-name> <message-type>
{% endhighlight %}

Some useful command-line flags:

- `-r/--rate <rate>` publishes this many times per second
- `-t/--times <times>` only runs this many times

<details>
<summary>Sample call to <code>turtlesim_node</code></summary>
<div>
{% highlight bash %}
ros2 topic pub -r 0.5 /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -3.1415926535}}"
{% endhighlight %}
</div>
</details>

#### Messages

A *message* is just a type of topic, which tells ROS what kind of data is being passed around. This is so that data can be encoded and decoded correctly.

### Service

A *service* in ROS is a way for [nodes](#node) to communicate with each other using call-and-response semantics. A node will send a request to another node, and receive a response in return. You can list available services using:

{% highlight bash %}
ros2 service list
{% endhighlight %}

<details>
<summary>Sample output when running <code>turtlesim_node</code></summary>
<div>
{% highlight bash %}
$ ros2 service list
/clear
/kill
/reset
/spawn
/turtle1/set_pen
/turtle1/teleport_absolute
/turtle1/teleport_relative
/turtlesim/describe_parameters
/turtlesim/get_parameter_types
/turtlesim/get_parameters
/turtlesim/list_parameters
/turtlesim/set_parameters
/turtlesim/set_parameters_atomically
{% endhighlight %}
</div>
</details>

### Action

A *action* in ROS has three parts:

1. Goal: A [service](#service) which can be called to kick off the action
2. Feedback: A [topic](#topic) which lets the actor give constant feedback to the controller
3. Result: A long-running [service](#service) which ultimately sends a response when the action has finished

Actions are essentially [services](#service) which can be pre-empted. You can list available actions using:

{% highlight bash %}
ros2 action list
{% endhighlight %}

<details>
<summary>Sample output when running <code>turtlesim_node</code></summary>
<div>
{% highlight bash %}
$ ros2 action list
/turtle1/rotate_absolute
{% endhighlight %}
</div>
</details>

### `run`

`ros2 run` is used to run a single executable (simply a [node](#node)). This can be invoked using:

{% highlight bash %}
ros2 run <package-name> <node-name>
{% endhighlight %}

For example, to run Turtlesim, use:

{% highlight bash %}
ros2 run turtlesim turtlesim_node
{% endhighlight %}

This will then print the logged output from the running process.

### `launch`

`ros2 launch` is used to run multiple [nodes](#node) at once, as defined by a launch file. A launch file can be written in Python, XML or YAML. For example, you can create a launch file in Python that looks like this:

{% highlight python %}
import launch_ros.actions
from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            launch_ros.actions.Node(
                namespace="turtlesim1",
                package="turtlesim",
                executable="turtlesim_node",
                output="screen",
            ),
            launch_ros.actions.Node(
                namespace="turtlesim2",
                package="turtlesim",
                executable="turtlesim_node",
                output="screen",
            ),
        ]
    )
{% endhighlight %}

If you save this somewhere (for example, `launch_turtlesim.py`) you can then run it using:

{% highlight bash %}
ros2 launch launch_turtlesim.py
{% endhighlight %}

In another terminal, you can then see two running Turtlesim nodes, each in their own respective namespace:

{% highlight bash %}
$ ros2 node list
/turtlesim1/turtlesim
/turtlesim2/turtlesim
{% endhighlight %}

### `colcon`

[Colcon][colcon-info] is the command line tool that ROS uses for building and testing. To build a Colcon project, you can run:

{% highlight bash %}
colcon build
{% endhighlight %}

This looks at each subdirectory in whichever directory you're in, checks if it is a ROS directory, and if so builds it. There are additional command-line options for choosing which subdirectories to include and ignore, such as:

- `--packages-select <pkg-1> (<pkg-2> ...)` Include these packages
- `--package-skip <pkg-1> (<pkg-2> ...)` Skip these packages

To run Colcon tests, you can use:

{% highlight bash %}
colcon test
{% endhighlight %}

After finishing, you can see any failures by running:

{% highlight bash %}
colcon test-result            # To show the test result summary files
colcon test-result --verbose  # To show all errors
{% endhighlight %}

## More Resources

Some other resources that I found helpful are listed below.

- [Ubuntu - Getting Started with ROS 2][ubuntu-ros-2-tutorial]: A quick introduction to the commands required to get started using ROS 2
- [ROS 2 Foxy Turtlesim Documentation][ros-2-foxy-turtlesim]: A more in-depth explanation of the different components of ROS 2 using Turtlesim
- [RoboStack Tutorial][robostack-tutorial-pdf]: In-depth tutorial covering RoboStack and various elements of ROS

[miniconda-install]: https://docs.conda.io/en/latest/miniconda.html
[stretch-robot]: https://hello-robot.com/stretch-2
[jetbot]: https://jetbot.org/master/
[robostack]: https://robostack.github.io/
[mamba]: https://anaconda.org/conda-forge/mamba
[ros-2-humble-install-instructions]: https://github.com/RoboStack/ros-humble
[ubuntu-ros-2-tutorial]: https://ubuntu.com/tutorials/getting-started-with-ros-2#1-overview
[ros-2-foxy-turtlesim]: https://docs.ros.org/en/foxy/Tutorials/Beginner-CLI-Tools/Introducing-Turtlesim/Introducing-Turtlesim.html
[turtlesim-ros-page]: http://wiki.ros.org/turtlesim
[tree-install-instructions]: https://www.cyberciti.biz/faq/linux-show-directory-structure-command-line/
[colcon-info]: https://colcon.readthedocs.io/en/released/
[robostack-tutorial-pdf]: https://arxiv.org/abs/2104.12910
