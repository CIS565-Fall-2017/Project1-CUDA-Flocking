**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Fengkai Wu
* Tested on: Windows 10, i7-6700 @ 3.40GHz 16GB, Quadro K620 4095MB (Moore 100C Lab)

## 1. Final Results

![test img](https://github.com/wufk/Project1-CUDA-Flocking/blob/master/images/Capture.PNG)

The simulation above is run under the following condition:

Number of Boids: 7500

Running Mode: unifor grid

Threads: (128, 1, 1)

Time step: 0.5


## 2. Analysis of Performance
### Algorithms

Following illustrate the performance running under different algorthms and number of Boids. The x-axis is the number of boids in the system and the y-axis is the simulation time of each step of computation.

![img_2](https://github.com/wufk/Project1-CUDA-Flocking/blob/master/images/mode.PNG)

It is clearly shown that using brute force to solve the problem takes huge amount of time and the average time increases drastically. Since the brute force does not optimize at all and compute every single element, the number of threads increases so fast that it is very difficult to parallize. However if we use unform grid method, the particles we calculate at each step is only constrained in certain cells, thus decreasing the amount of threads for simulation, thus decreasing the average time.

As for the scattered mode and coherent mode, though the difference of these two is trivial, the coherent mode still outperforms. This is because by sacrificing memory to maintaining a new buffer array for position, we can save the time of looking up boid original positions and velocities. 

### Other factors

The output of the folloing table is runned using uniform grid. 

![img_3](https://github.com/wufk/Project1-CUDA-Flocking/blob/master/images/other%20factors.PNG)

From the table, we can see some obvious results. The max speed and dT has little to do with performance because they do not affect the simulation. A larger number of neighboring cells also increase the time because of more computation per step. 

