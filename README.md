**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 0**

* Mauricio Mutai
* Tested on: Windows 10, i7-7700HQ @ 2.80GHz 16GB, GTX 1050Ti 4GB (Personal Computer)

### Overview

#### Introduction

The main aim of this project was to get myself acquaintanced with CUDA programming. After learning some basic CUDA concepts in class, I implemented this boid simulation in order to leverage the parallel computing capabilities of a GPU.

Three different search algorithms (naive, uniform grid, coherent grid) were implemented in order to investigate which provided the best performance for the simulation. In addition, smaller tweaks to these algorithms, such as CUDA block size, were tested as well.

Below are images of the final working project. More specifically, it is running the coherent grid search with 10000 boids, block size of 128, and cell width of two neighor distances.

#### Screenshot of boid simulation in progress

![](images/boids-still.png)

#### Animation of boid simulation in progress

![](images/boids-anim.gif)

### Performance Analysis

#### Methodology

I measured the performance of the simulation by using the frames per second (FPS) metric displayed on the window's title. I found that it fluctuated too much to give consistent readings, so I added the average of the last 10 FPS measurements to the window title as well.

When measuring FPS, I tried to keep the running environment as consistent as possible. For example, I noticed that having Chrome open directly under the running simulation led to significantly lower framerates than when Visual Studio was open in that position. Thus, I always measured with Visual Studio open, and waited at least 10 seconds for each measurement in order to have a more stable and consistent FPS reading.

Below, I examine the effect of several parameters of the simulation on its performance.

#### Number of boids

Below is a graph showing how FPS changes for the various search algorithms as the number of boids increases.

![](images/graphNumBoids.png)
