**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Ricky Rajani
* Tested on: Windows 7, i7-6700 @ 3.40GHz 16GB, NVIDIA Quadro K620 (Moore 100C Lab)

**5,000 Boids on Coherent Uniform Grid**

![](images/perf-analysis/Shot1.PNG)

**50,000 Boids on Coherent Uniform Grid**

![](images/perf-analysis/Shot2.PNG)

**100,000 Boids on Coherent Uniform Grid**

![](images/perf-analysis/Shot3.PNG)

### Performance Analysis

![](images/perf-analysis/Graph1.PNG)

![](images/perf-analysis/Graph2.PNG)

For each implementation, how does changing the number of boids affect performance?

As the number of boids increased, there was a significant drop in performance for each implementation. However, the scattered uniform grid method and coherent uniform grid method improved performance considerably. Improvement in performance can be attributed to runtime complexity of the search algorithms in each implementation. The uniform grid method decreased the number of boids that were checked during each iteration. On top of that, coherence made memory access significantly faster.

For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid?

By rearranging the boid data such that velocities and positions of boid could be accessed quickly, there was a considerable performance improvement. This method allowed for direct access to boid data that worked around the GPU having to jump around in memory via pointers.

![](images/perf-analysis/Graph3.PNG)

![](images/perf-analysis/Graph4.PNG)

For each implementation, how does changing the block count and block size affect performance?

All implementations show a significant performance hit at low block sizes, specifically 32, and a slight decrease in performance as blocksize increased past 128 towards the max value of 1024. For smaller block sizes, it should be noted that there can only be one warp (32 threads) in a block. This leads to a decrease in shared memory and a need for more blocks. For large block sizes, there is a need to decrease the number of blocks we can have because there is a cap on the number of threads the GPU can handle.
