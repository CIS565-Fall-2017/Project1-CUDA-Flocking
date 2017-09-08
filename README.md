**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Yalun Hu
* Tested on: Windows 10, i7-6700HQ CPU @ 2.60GHz 32GB, GTX 1070 8192MB (Personal computer)

### Flocking

![](images/flock.gif)

### Features

* Naive Boids Simulation
* Scattered uniform grids Boids Simulation
* Coherent uniform grids Boids Simulation

### Performance Analysis

Methods for measuring performance:

* **Framerate change:**  

* **Time of step simulation:**


## Performance plots

# Framerate change with increasing # of boids

* with visualization

<p align="center">
  <img src="images/fps-boids-visual">
</p>

* without visualization

<p align="center">
  <img src="images/fps-boids-nonvisual">
</p>

# Framerate change with increasing block size

<p align="center">
  <img src="images/fps-block-nonvisual.png">
</p>

# Step simulation time with increasing # of boids

* with visualization

<p align="center">
  <img src="images/time-boids-visual.png">
</p>

* without visualization

<p align="center">
  <img src="images/time-boids-nonvisual.png">
</p>

# Step simulation time with increasing block size

<p align="center">
  <img src="images/time-block-nonvisual.png">
</p>

## Questions and answers

* **For each implementation, how does changing the number of boids affect performance? Why do you think this is?**



* **For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**



* **For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**



* **Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?**
