Project 1 CUDA Flocking
====================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

* Mariano Merchante
* Tested on
  * Microsoft Windows 10 Pro
  * Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz, 2601 Mhz, 4 Core(s), 8 Logical Processor(s)
  * 32.0 GB RAM
  * NVIDIA GeForce GTX 1070 (mobile version)

## Details

This project solves a flocking simulation with three different implementations: a brute force approach, a uniform grid and a coherent version of this grid. It handles arbitrary cell size for the uniform grid, and tries to be as precise as possible when checking neighbor cells to prevent iterating unnecessary particles.

## Analysis

For evaluating performance, I used time per frame in milliseconds and disabled the visualization, so that there's no overhead from the actual rendering pipeline.

* For each implementation, how does changing the number of boids affect performance? Why do you think this is?
  * For the brute force approach performance drops very quickly, because it scales with the square of the particle count. This happens due to the nature of iterating over every particle when looking for neighbors.
  * For the uniform grid approach, there's a slight initial overhead because of all the bookkeeping needed to maintain the grid, but it scales linearly over particle count.
  * For the coherent grid approach, there's no obvious improvement in performance -- and actually it decreases a bit compared to the uniform grid. A reason for this could be that both implementations still access global memory and thus any improvement will depend on L2 size and speed. Also, the random access shuffle for particle indices every frame may impact performance more than the access of the neighbor cells' particles. With shared memory this approach should be considerably faster, however.
