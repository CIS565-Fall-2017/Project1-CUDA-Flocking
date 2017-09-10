# University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1 - Flocking

* Qiaoyu Zhang
* Tested on:
  * Windows 10, i7-6700 @ 3.40GHz 64GB, GTX 1070 8192MB (Personal Desktop)
  * Ubuntu 16.04, i7-6700HQ @ 3.50GHz 16GB, GTX 1060 6144MB (Personal Laptop)

## Summary

This project implements a flocking simulation based on the [Reynolds Boids algorithm](http://www.red3d.com/cwr/boids/) on a GPU using CUDA and OpenGL, along with two levels of optimization: a univorm grid, and a uniform grid with semi-coherent memory access.

## Result