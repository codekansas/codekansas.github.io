---
layout: post
title: PID Controller Explainer
tags: [eng, algos, math]
excerpt: >
  Simple overview of what a PID controller is, how it works, and how to make one yourself.
---

{% katexmm %}

I was recently trying to explain PID controllers to someone and realized that I didn't have a very good intuitive understanding of what they're useful for and how they work. When looking around the web, I had trouble finding a straightforward explainer. So in this post, I'll give (hopefully) simple answers to some basic questions that I had about PID controllers.

## What is a PID controller?

A PID controller is a way to solve problems with the following formulation:

- You can change some input to the system, called the process variable
- You have a sensor which monitors something about the system
- You want the sensor measurement to be close to some target value, called the set point

The PID controller is a good way to decide what the input to the system should be without knowing anything about the internal workings of the system, except that the change in output is roughly proportional to the input.

### Example Use Cases

#### 3D Printing

When running a 3D printer, you want the nozzel end to be a specific temperature. You control the temperature by regulating the voltage through a hot end - higher voltage makes the temperature go up, whereas if you turn it off entirely the temperature will go down (sometimes helped by a fan next to the extruder). You likely want to change the temperature during the course of printing, and you want it to reach the target temperature as quickly as possible. Changing the voltage affects the _rate of change in temperature_ rather than the temperature itself.

#### Vehicle Control

When driving a car, you regulate the speed by controlling how much to open the throttle. Opening the throttle will cause the car to accelerate, while closing it will cause the car to decelerate. You want the car to reach some set speed as quickly as possible. Changing the throttle toggles the _acceleration_ and not the _velocity_, but the variable you care about controlling is the _velocity_.

#### Medicine

When giving vasopressors to a patient in a hospital, you want them to reach some target blood pressure. After injecting a particular amount, the patient's blood pressure will go up or down. Changing that amount will change the _rate of change_ of their blood pressure. You want to reach the target blood pressure as quickly as possible without overshooting.

## How do PID controllers work?

To answer this question, I think the best way is to start off with an easy-to-understand controller, then add on top of it until we get to the final PID formulation.

### A Simple Controller

You could write a control rule like this:

- If the sensor measurement is too low, set the system input to "positive" (try to make the sensor measurement higher)
- If the sensor measurement is too high, set the system input to "negative" (try to make the sensor measurement lower)

However, if the system has _inertia_ (in other words, a delay between the change in input and the change in output), then this control algorithm will start oscillating as you repeatedly undershoot and overshoot. Inertia can happen in lots of different ways and is common in most systems that you would actually want to control. An improvement to this could be to scale the input relative to the _error_, so that as your error gets smaller, you decrease your input.

$$
\begin{aligned}
\text{error} & = \text{target} - \text{sensor} \\
\text{input} & = K_p * \text{error}
\end{aligned}
$$

In this case, we introduce an additional scaling constant $K_p$, which relates the size of the error to the size of the input. For example, in the case of our 3D printer, our error is the difference between the target temperature and the observed temperature, while our input is the voltage, so we need to convert from degrees celcius to volts somehow.

We can try these out with some sample numbers to demonstrate the idea. Suppose our target temperature is $200 \deg$ and our current temperature is $175 \deg$. We can use some scaling constant $K_p = 0.1 V / \deg$.

$$
\begin{aligned}
\text{error} & = 200 \deg - 175 \deg = 25 \deg \\
\text{input} &= (0.1 V / \deg) * 25 \deg = 2.5 V
\end{aligned}
$$

After some time, the temperature increases to $190 \deg$.

$$
\begin{aligned}
\text{error} & = 200 \deg - 190 \deg = 10 \deg \\
\text{input} &= (0.1 V / \deg) * 10 \deg = 1 V
\end{aligned}
$$

As expected, our input is smaller than when the error was larger, since we want to make the temperature delta smaller as we get closer to our target temperature.

Let's say a bit later the system has gotten hotter, and now our reading is $240 \deg$.

$$
\begin{aligned}
\text{error} & = 200 \deg - 240 \deg = -40 \deg \\
\text{input} &= (0.1 V / \deg) * -40 \deg = -4 V
\end{aligned}
$$

As expected, now that we've overshot our target temperature, we need to supply an input in the opposite direction. Note that the input voltage is relative to some zero point, since we can't actually have a negative input voltage.

### Using this Controller

Here's a simple program to simulate a 3D printer nozzel. Note that the heater simulator is only a very loose approximation to the behavior of the actual heater, and can stand in for any black box system that you might want to use a PID controller for.

```python
import argparse

import matplotlib.pyplot as plt

class SimpleController:
    def __init__(self, trg_temp: float, kp: float, v_offset: float) -> None:
        """Initializes a simple controller.

        Args:
            trg_temp: The target temperature
            kp: The P scaling constant
            v_offset: The zero point voltage
        """

        self.trg_temp = trg_temp
        self.kp = kp
        self.v_offset = v_offset

    def step(self, temperature: float) -> float:
        """Gets the target voltage for the current timestep.

        Args:
            temperature: The last observed temperature

        Returns:
            The input voltage for the next timestep
        """

        error = self.trg_temp - temperature
        return self.kp * error + self.v_offset

class HeaterSimulator:
    def __init__(
        self,
        dt: float,
        amb_temp: float,
        min_voltage: float,
        max_voltage: float,
        heat_coeff: float,
        area: float,
        voltage_coeff: float,
        inertia: float,
    ) -> None:
        """Initializes the heater simulator.

        Args:
            dt: The timestep size, in seconds
            amb_temp: The ambient temperature
            min_voltage: The minimum input voltage
            max_voltage: The maximum input voltage
            heat_coeff: Heat transfer coefficient
            area: Heater surface area
            voltage_coeff: The voltage-to-temperature-delta coefficient
            inertia: System inertia
        """

        self.dt = dt
        self.amb_temp = amb_temp
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.heat_coeff = heat_coeff
        self.area = area
        self.voltage_coeff = voltage_coeff
        self.inertia = inertia

        # Heater temperatures starts at the ambient temperature.
        self.temperature = amb_temp

        # Add some inertia to dtemp.
        self.dtemp = 0.0

    def step(self, voltage: float) -> None:
        """Runs the simulator for one step.

        Args:
            voltage: The input voltage
        """

        q_rate = self.heat_coeff * self.area * (self.amb_temp - self.temperature)
        v_rate = self.voltage_coeff * voltage
        trg_dtemp = q_rate + v_rate

        self.dtemp = self.inertia * self.dtemp + (1 - self.inertia) * trg_dtemp
        self.temperature += self.dtemp * self.dt

def main() -> None:
    parser = argparse.ArgumentParser(description="Heater PID simulation")
    parser.add_argument("--kp", type=float, nargs="+", required=True, help="P scale")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep size")
    parser.add_argument("--total-steps", type=int, default=10000, help="Number of simulation steps")
    parser.add_argument("--amb-temp", type=float, default=20.0, help="Ambient temperature")
    parser.add_argument("--trg-temp", type=float, default=210.0, help="Target temperature")
    parser.add_argument("--v-offset", type=float, default=3.0, help="Offset voltage")
    parser.add_argument("--min-voltage", type=float, default=0.0, help="Minimum voltage")
    parser.add_argument("--max-voltage", type=float, default=24.0, help="Maximum voltage")
    parser.add_argument("--area", type=float, default=1e-4, help="Surface area of the heater")
    parser.add_argument("--heat-coeff", type=float, default=100.0, help="Heat transfer coefficient")
    parser.add_argument("--voltage-coeff", type=float, default=1.0, help="Voltage coefficient")
    parser.add_argument("--inertia", type=float, default=0.99, help="System inertia")
    args = parser.parse_args()

    # Plots the simulated temperatures.
    plt.figure()

    for kp in args.kp:
        # Simulator.
        simulator = HeaterSimulator(
            dt=args.dt,
            amb_temp=args.amb_temp,
            min_voltage=args.min_voltage,
            max_voltage=args.max_voltage,
            heat_coeff=args.heat_coeff,
            area=args.area,
            voltage_coeff=args.voltage_coeff,
            inertia=args.inertia,
        )

        # Controller.
        controller = SimpleController(
            trg_temp=args.trg_temp,
            kp=kp,
            v_offset=args.v_offset,
        )

        # Heater starts at ambient temperature.
        temperatures = [simulator.temperature]
        voltages = [0.0]
        times = [i * args.dt for i in range(1, args.total_steps + 1)]

        for _ in times:
            voltage = controller.step(simulator.temperature)
            voltages.append(voltage)
            simulator.step(voltage)
            temperatures.append(simulator.temperature)

        times = [0.0] + times

        # Plot temperature.
        plt.plot(times, temperatures, label=f"Kp = {kp:.4g}")
        plt.ylabel("Temperature")

        # Plot voltages.
        # plt.plot(times, voltages, label=f"Kp = {kp:.4g}")
        # plt.ylabel("Voltage")

    plt.xlabel("Time")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
```

Here is just the controller, without the boilerplate for running it.

```python
class SimpleController:
    def __init__(self, trg_temp: float, kp: float, v_offset: float) -> None:
        """Initializes a simple controller.

        Args:
            trg_temp: The target temperature
            kp: The P scaling constant
            v_offset: The zero point voltage
        """

        self.trg_temp = trg_temp
        self.kp = kp
        self.v_offset = v_offset

    def step(self, temperature: float) -> float:
        """Gets the target voltage for the current timestep.

        Args:
            temperature: The last observed temperature

        Returns:
            The input voltage for the next timestep
        """

        error = self.trg_temp - temperature
        return self.kp * error + self.v_offset
```

We can run this script using:

```bash
python simulator.py --kp 0.05 0.1 0.2 0.4 0.8
```

You can try running this script yourself to see how playing with different parts of the system affect the temperature curves. In particular, `--v-offset` and `--inertia` are interesting parameters to play with.

The resulting temperature curves for the built-in configuration are as follows:

{% include /images/pid-controller/simple_controller.svg %}

### Proportional, Integral, Derivative

PID stands for "proportional, integral, derivative" and is a way to address some issues with the above model. Namely, there are two issues that we want to address:

1. Undershoot: Our input is too weak, and the output isn't changing quickly enough in response to a change in input
2. Overshoot: Our input is too strong, and the output is changing too quickly

Among the temperature curves above, the $K_p = 0.8$ overshoots the most, while the $K_p = 0.05$ undershoots the most.

### Integral Control

Consider the case in which we are **undershooting** our target value. We can detect that we're undershooting if the error is accumulating too fast - in other words, our error isn't going down fast enough. We can add another term to the controller which takes this into account, using the integral of the error.

$$\text{input} = K_i \int \text{error}_t \, d \text{ time}$$

We can approximate this by keeping track of our running error:

$$\text{input} \approx K_i \sum_{t = 0}^{T} \text{error}_t \, d \text{ time}$$

This controller can work on its own, and will correct for undershooting. However, by itself it will naturally oscillate, because it has to accumulate error on the opposite side of the target value in order to start heading in the opposite direction.

Here's a few temperature curves for an undershooting proportional controller, with different integral controller coefficients.

{% include /images/pid-controller/integral_controller.svg %}

### Derivative Control

Consider the case in which we are **overshooting** our target value. We can detect that we're about to overshoot if the error is getting smaller too fast. We can add another term to the controller which takes this into account, using the derivative of the error. The desired behavior is to decrease the input if the error is getting smaller too quickly, and increase the input if the error is getting smaller too slowly. This can be expressed as a function of the derivative of the error:

$$\text{input} = K_d \frac{d \, \text{error}}{d \, \text{time}}$$

We can approximate this by keeping track of our past error:

$$\text{input} \approx K_d \frac{\text{error} - \text{prev error}}{d \, \text{time}}$$

This controller won't work on its own, because the error shouldn't change without changing the input. The power of this controller is to help correct for the overshooting behavior of our original controller.

Here's a few temperature curves for an overshooting proportional controller, with different derivative controller coefficients.

{% include /images/pid-controller/derivative_controller.svg %}

### Updating our Controller

We can put together each of our controllers into the final PID controller formulation shown below:
$$\text{input} = K_p \text{error} + K_i \text{error} \, d \text{ time} + K_d \frac{d \text{error}}{d \, \text{time}}$$
A Python implementation for this controller can be found below.

```python
import argparse
import itertools
from typing import Optional

import matplotlib.pyplot as plt

class PIDController:
    def __init__(
        self,
        dt: float,
        trg_temp: float,
        kp: float,
        ki: float,
        kd: float,
        v_offset: float,
    ) -> None:
    """Initializes a simple controller.

        Args:
            dt: The timestep size, in seconds
            trg_temp: The target temperature
            kp: The P scaling constant
            ki: The I scaling constant
            kd: The D scaling constant
            v_offset: The zero point voltage
        """

        self.dt = dt
        self.trg_temp = trg_temp
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.v_offset = v_offset

        self.prev_error: Optional[float] = None
        self.acc_error: Optional[float] = None

    def step(self, temperature: float) -> float:
        """Gets the target voltage for the current timestep.

        Args:
            temperature: The last observed temperature

        Returns:
            The input voltage for the next timestep
        """

        error = self.trg_temp - temperature

        # Proportional control.
        p_term = self.kp * error

        # Integral control.
        acc_error = error if self.acc_error is None else error + self.acc_error
        self.acc_error = acc_error
        i_term = self.ki * acc_error

        # Derivative control.
        delta_error = 0.0 if self.prev_error is None else error - self.prev_error
        self.prev_error = error
        d_term = self.kd * delta_error

        return p_term + i_term + d_term + self.v_offset

class HeaterSimulator:
    def __init__(
        self,
        dt: float,
        amb_temp: float,
        min_voltage: float,
        max_voltage: float,
        heat_coeff: float,
        area: float,
        voltage_coeff: float,
        inertia: float,
    ) -> None:
        """Initializes the heater simulator.

        Args:
            dt: The timestep size, in seconds
            amb_temp: The ambient temperature
            min_voltage: The minimum input voltage
            max_voltage: The maximum input voltage
            heat_coeff: Heat transfer coefficient
            area: Heater surface area
            voltage_coeff: The voltage-to-temperature-delta coefficient
            inertia: System inertia
        """

        self.dt = dt
        self.amb_temp = amb_temp
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.heat_coeff = heat_coeff
        self.area = area
        self.voltage_coeff = voltage_coeff
        self.inertia = inertia

        # Heater temperatures starts at the ambient temperature.
        self.temperature = amb_temp

        # Add some inertia to dtemp.
        self.dtemp = 0.0

    def step(self, voltage: float) -> None:
        """Runs the simulator for one step.

        Args:
            voltage: The input voltage
        """

        q_rate = self.heat_coeff * self.area * (self.amb_temp - self.temperature)
        v_rate = self.voltage_coeff * voltage
        trg_dtemp = q_rate + v_rate

        self.dtemp = self.inertia * self.dtemp + (1 - self.inertia) * trg_dtemp
        self.temperature += self.dtemp * self.dt

def main() -> None:
    parser = argparse.ArgumentParser(description="Heater PID simulation")
    parser.add_argument("--kp", type=float, nargs="+", required=True, help="P scale")
    parser.add_argument("--ki", type=float, nargs="+", required=True, help="I scale")
    parser.add_argument("--kd", type=float, nargs="+", required=True, help="D scale")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep size")
    parser.add_argument("--total-steps", type=int, default=10000, help="Number of simulation steps")
    parser.add_argument("--amb-temp", type=float, default=20.0, help="Ambient temperature")
    parser.add_argument("--trg-temp", type=float, default=200.0, help="Target temperature")
    parser.add_argument("--v-offset", type=float, default=3.0, help="Offset voltage")
    parser.add_argument("--min-voltage", type=float, default=0.0, help="Minimum voltage")
    parser.add_argument("--max-voltage", type=float, default=24.0, help="Maximum voltage")
    parser.add_argument("--area", type=float, default=1e-4, help="Surface area of the heater")
    parser.add_argument("--heat-coeff", type=float, default=100.0, help="Heat transfer coefficient")
    parser.add_argument("--voltage-coeff", type=float, default=1.0, help="Voltage coefficient")
    parser.add_argument("--inertia", type=float, default=0.99, help="System inertia")
    args = parser.parse_args()

    # Plots the simulated temperatures.
    plt.figure()

    for kp, ki, kd in itertools.product(args.kp, args.ki, args.kd):
        # Simulator.
        simulator = HeaterSimulator(
            dt=args.dt,
            amb_temp=args.amb_temp,
            min_voltage=args.min_voltage,
            max_voltage=args.max_voltage,
            heat_coeff=args.heat_coeff,
            area=args.area,
            voltage_coeff=args.voltage_coeff,
            inertia=args.inertia,
        )

        # Controller.
        controller = PIDController(
            dt=args.dt,
            trg_temp=args.trg_temp,
            kp=kp,
            ki=ki,
            kd=kd,
            v_offset=args.v_offset,
        )

        # Heater starts at ambient temperature.
        temperatures = [simulator.temperature]
        voltages = [0.0]
        times = [i * args.dt for i in range(1, args.total_steps + 1)]

        for _ in times:
            voltage = controller.step(simulator.temperature)
            voltages.append(voltage)
            simulator.step(voltage)
            temperatures.append(simulator.temperature)

        times = [0.0] + times

        # Plot temperature.
        plt.plot(times, temperatures, label=f"Kp = {kp:.4g}, Ki = {ki:.4g}, Kd = {kd:.4g}")
        plt.ylabel("Temperature")

        # Plot voltages.
        # plt.plot(times, voltages, label=f"Kp = {kp:.4g}")
        # plt.ylabel("Voltage")

    plt.xlabel("Time")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
```

The snippet below has just the code for the controller, without the boilerplate for running it.

```python
class PIDController:
    def __init__(
        self,
        dt: float,
        trg_temp: float,
        kp: float,
        ki: float,
        kd: float,
        v_offset: float,
    ) -> None:
        """Initializes a simple controller.

        Args:
            dt: The timestep size, in seconds
            trg_temp: The target temperature
            kp: The P scaling constant
            ki: The I scaling constant
            kd: The D scaling constant
            v_offset: The zero point voltage
        """

        self.dt = dt
        self.trg_temp = trg_temp
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.v_offset = v_offset

        self.prev_error: Optional[float] = None
        self.acc_error: Optional[float] = None

    def step(self, temperature: float) -> float:
        """Gets the target voltage for the current timestep.

        Args:
            temperature: The last observed temperature

        Returns:
            The input voltage for the next timestep
        """

        error = self.trg_temp - temperature

        # Proportional control.
        p_term = self.kp * error

        # Integral control.
        acc_error = error if self.acc_error is None else error + self.acc_error
        self.acc_error = acc_error
        i_term = self.ki * acc_error

        # Derivative control.
        delta_error = 0.0 if self.prev_error is None else error - self.prev_error
        self.prev_error = error
        d_term = self.kd * delta_error

        return p_term + i_term + d_term + self.v_offset

```

## How Can you Make One Yourself?

Now that we've figured out the basic formulation for PID controllers, how can we figure out the values for $K_p$, $K_I$ and $K_d$ which give us the best behavior?

There are a few methods for doing this, and it depends a lot on the particular scenario. Some relevant questions:

- Can you run the control algorithm many times, or is it important to run it only a few times?
- Is there a lot of noise in the system, or if you run the same control algorithm with the same parameters will the result be relatively similar each time?

There are many packages and techniques which will figure out these parameters for you, but if you're in a pinch you can follow the approach below.

1. Choose an objective to minimize
2. Do a grid search over $K_p$, $K_i$ and $K_d$
3. Choose the values of $K_p$, $K_i$ and $K_d$ which minimize that objective

### Define the Objective to Minimize

For most PID controllers, you care about making the output reach the set point as quickly as possible, without overshoot. Different applications can tolerate different amount of overshoot and undershoot, so the metric might vary. However, in this application I'm going to choose a simple error function which optimizes for both quickly reaching the target and not overshooting, by taking the absolute error when the output is less than the set point and the squared error when the output is greater than the set point.

$$
\begin{aligned}
\text{error} & = \text{set point} - \text{output} \\
L & = \int_0^T
\begin{cases}
  \text{error}_t^2 & \text{if } \text{error}_t < 0 \\
  \text{error}_t & \text{if } \text{otherwise}
\end{cases}
\end{aligned}
dt
$$

### Do a Grid Search over $K_p$, $K_i$ and $K_d$

I've included a script which can be used for sweeping different PID configurations for our original simulation.

```python
import argparse
import itertools
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

class PIDController:
    def __init__(
        self,
        dt: float,
        trg_temp: float,
        kp: float,
        ki: float,
        kd: float,
        v_offset: float,
    ) -> None:
        """Initializes a simple controller.

        Args:
            dt: The timestep size, in seconds
            trg_temp: The target temperature
            kp: The P scaling constant
            ki: The I scaling constant
            kd: The D scaling constant
            v_offset: The zero point voltage

        """

        self.dt = dt
        self.trg_temp = trg_temp
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.v_offset = v_offset

        self.prev_error: Optional[float] = None
        self.acc_error: Optional[float] = None

    def step(self, temperature: float) -> Tuple[float, float]:
        """Gets the target voltage for the current timestep.

        Args:
            temperature: The last observed temperature

        Returns:
            The input voltage for the next timestep
        """

        error = self.trg_temp - temperature

        # Proportional control.
        p_term = self.kp * error

        # Integral control.
        acc_error = error if self.acc_error is None else error + self.acc_error
        self.acc_error = acc_error
        i_term = self.ki * acc_error

        # Derivative control.
        delta_error = 0.0 if self.prev_error is None else error - self.prev_error
        self.prev_error = error
        d_term = self.kd * delta_error

        return p_term + i_term + d_term + self.v_offset, error

class HeaterSimulator:
    def __init__(
        self,
        dt: float,
        amb_temp: float,
        min_voltage: float,
        max_voltage: float,
        heat_coeff: float,
        area: float,
        voltage_coeff: float,
        inertia: float,
    ) -> None:
        """Initializes the heater simulator.

        Args:
            dt: The timestep size, in seconds
            amb_temp: The ambient temperature
            min_voltage: The minimum input voltage
            max_voltage: The maximum input voltage
            heat_coeff: Heat transfer coefficient
            area: Heater surface area
            voltage_coeff: The voltage-to-temperature-delta coefficient
            inertia: System inertia
        """

        self.dt = dt
        self.amb_temp = amb_temp
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.heat_coeff = heat_coeff
        self.area = area
        self.voltage_coeff = voltage_coeff
        self.inertia = inertia

        # Heater temperatures starts at the ambient temperature.
        self.temperature = amb_temp

        # Add some inertia to dtemp.
        self.dtemp = 0.0

    def step(self, voltage: float) -> None:
        """Runs the simulator for one step.

        Args:
            voltage: The input voltage
        """

        q_rate = self.heat_coeff * self.area * (self.amb_temp - self.temperature)
        v_rate = self.voltage_coeff * voltage
        trg_dtemp = q_rate + v_rate

        self.dtemp = self.inertia * self.dtemp + (1 - self.inertia) * trg_dtemp
        self.temperature += self.dtemp * self.dt

def main() -> None:
    parser = argparse.ArgumentParser(description="Heater PID simulation")
    parser.add_argument("--kp", type=float, nargs="+", required=True, help="P scale")
    parser.add_argument("--ki", type=float, nargs="+", required=True, help="I scale")
    parser.add_argument("--kd", type=float, nargs="+", required=True, help="D scale")
    parser.add_argument("--plot", type=str, choices=["kp", "ki", "kd"], required=True, help="Which to plot")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep size")
    parser.add_argument("--total-steps", type=int, default=10000, help="Number of simulation steps")
    parser.add_argument("--amb-temp", type=float, default=20.0, help="Ambient temperature")
    parser.add_argument("--trg-temp", type=float, default=200.0, help="Target temperature")
    parser.add_argument("--v-offset", type=float, default=3.0, help="Offset voltage")
    parser.add_argument("--min-voltage", type=float, default=0.0, help="Minimum voltage")
    parser.add_argument("--max-voltage", type=float, default=24.0, help="Maximum voltage")
    parser.add_argument("--area", type=float, default=1e-4, help="Surface area of the heater")
    parser.add_argument("--heat-coeff", type=float, default=100.0, help="Heat transfer coefficient")
    parser.add_argument("--voltage-coeff", type=float, default=1.0, help="Voltage coefficient")
    parser.add_argument("--inertia", type=float, default=0.99, help="System inertia")
    args = parser.parse_args()

    def error_func(error: float) -> float:
        return error if error > 0 else error**2

    def get_error(kp: float, ki: float, kd: float) -> float:
        # Simulator.
        simulator = HeaterSimulator(
            dt=args.dt,
            amb_temp=args.amb_temp,
            min_voltage=args.min_voltage,
            max_voltage=args.max_voltage,
            heat_coeff=args.heat_coeff,
            area=args.area,
            voltage_coeff=args.voltage_coeff,
            inertia=args.inertia,
        )

        # Controller.
        controller = PIDController(
            dt=args.dt,
            trg_temp=args.trg_temp,
            kp=kp,
            ki=ki,
            kd=kd,
            v_offset=args.v_offset,
        )

        times = [i * args.dt for i in range(1, args.total_steps + 1)]
        total_error = 0.0

        for _ in times:
            voltage, error = controller.step(simulator.temperature)
            simulator.step(voltage)
            total_error += error_func(error) * args.dt

        return total_error

    kps = args.kp
    kis = args.ki
    kds = args.kd
    index = 0 if args.plot == "kp" else 1 if args.plot == "ki" else 2

    def linspace(vals: List[float]) -> List[float]:
        assert len(vals) == 2, f"Expected `{args.plot}` to have exactly two items, not {len(vals)}"
        min_val, max_val = vals
        return [i * (max_val - min_val) / (args.num_samples - 1) + min_val for i in range(args.num_samples)]

    if index == 0:
        kcs, kas, kbs = ("Kp", linspace(kps)), ("Ki", kis), ("Kd", kds)
    elif index == 1:
        kcs, kas, kbs = ("Ki", linspace(kis)), ("Kp", kps), ("Kd", kds)
    else:
        kcs, kas, kbs = ("Kd", linspace(kds)), ("Kp", kps), ("Ki", kis)

    plt.figure()

    for ka, kb in itertools.product(kas[1], kbs[1]):
        values: List[float] = []
        errors: List[float] = []
        for kc in kcs[1]:
            vals = {kas[0]: ka, kbs[0]: kb, kcs[0]: kc}
            kp, ki, kd = vals["Kp"], vals["Ki"], vals["Kd"]
            values.append(kc)
            errors.append(get_error(kp, ki, kd))
        plt.plot(values, errors, label=f"{kas[0]}: {ka:.3g}, {kbs[0]}: {kb:.3g}")

    plt.xlabel(["Kp", "Ki", "Kd"][index])
    plt.ylabel("Error")
    plt.semilogy()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
```

The snippet below contains just the code for the controller, without the code for running it.

```python
def error_func(error: float) -> float:
    return error if error > 0 else error**2

def get_error(kp: float, ki: float, kd: float) -> float: # Simulator.
    simulator = HeaterSimulator(
        dt=args.dt,
        amb_temp=args.amb_temp,
        min_voltage=args.min_voltage,
        max_voltage=args.max_voltage,
        heat_coeff=args.heat_coeff,
        area=args.area,
        voltage_coeff=args.voltage_coeff,
        inertia=args.inertia,
    )

    # Controller.
    controller = PIDController(
        dt=args.dt,
        trg_temp=args.trg_temp,
        kp=kp,
        ki=ki,
        kd=kd,
        v_offset=args.v_offset,
    )

    times = [i * args.dt for i in range(1, args.total_steps + 1)]
    total_error = 0.0

    for _ in times:
        voltage, error = controller.step(simulator.temperature)
        simulator.step(voltage)
        total_error += error_func(error) * args.dt

    return total_error
```

The script can be run, for example, using the command below:

```bash
python sweep.py --kp 0.0 3.0 --ki 0.0 --kd 0.0 10.0 50.0 100.0 --plot kp
```

Running this command gives the following plot of the error curves generated when varying $K_p$ for different values of $K_d$:

{% include /images/pid-controller/pd_sweep.svg %}

As this graph shows, the ideal value of $K_p$ (the lowest point in the error curve) increases as we increase our value for $K_d$. This makes sense intuitively; by increasing the derivative term, the temperature will not shoot up as quickly for the same value of $K_p$, but we can safely use a higher $K_p$ without worrying about overshooting.

### Choose the values of $K_p$, $K_i$ and $K_d$ which minimize that objective

Now that we've looked at a few different configurations, we can just choose the configuration which minimizes our objective.

We can plot the associated temperature curve using the command below:

```bash
python printer.py --kp 1.65 --ki 0.0 --kd 100.0
```

It looks reasonable, and definitely better than our original curve.

{% include /images/pid-controller/pd_sweep_best.svg %}

We can do a much more careful job and get a better curve, but this is pretty reasonable for our toy problem. In fact, for the default parameters, we can just make $K_p$ and $K_d$ really large and get very close to an ideal curve. It's kind of fun to play around with different values for `--heat-coeff`, `--voltage-coeff` and `--inertia` to see how that changes the ideal PID parameters.

{% endkatexmm %}
