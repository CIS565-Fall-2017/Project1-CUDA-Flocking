**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Sarah Forcier
* Tested on: GeForce GTX 1070

| 100,000 boids | 5,000 boids | 
| ------------- | ----------- |
| ![](flocking.gif) | ![](flocking1.gif) |
| 720 FPS | 620 FPS |

### Performance Analysis

#### Framerate change with increasing number of boids
![](typecomparison.png)

#### Framerate change with and without visualization
![](visualization.png)

#### Framerate change with increasing block size
![](blocksize.png)

### Q&A

#### How does changing the number of boids affect performance? Why?
Generally, increasing the number of boids slows performance. However, the brute force method performs comparably with the grid-accelerated method for lower boid counts because iterating through all the boids can be completed faster than the necessary set up required for sorting the boids with respect to the grid.

#### How does changing the block count and block size affect performance? Why?
The performance increases with block size, but plateaus at 32 because the GPU architecture has a warp size of 32. Smaller block sizes do not take advantage of the full warp, but larger sizes that are multiples of 32 cannot get better performance because they have already use the full warp. However, for blocksizes that are not multiples of 32, performance is negatively affected because some warps will not be completely filled. 

#### For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
A speed up was achieved by using a coherent uniform grid. This is the expected behavior because it reduces the number of cache misses that occur when the position and velocity is looked up. In the uniform grid implementation, the boid pointers are sorted in grid order but these pointers point to uncontiguous memory. The coherent uniform grid sorts the velocity and position data directly so that these values are contiguous.  

#### Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?
Decreasing the cell width to the neighborhood distance, and checking 27 neighboring cells decreased the performace by a factor of 7 for the coherent and uniform grid methods. The brute force method is not affected by this change. The simulation is much slower because there are simply more cells to check during each step, and with more cells to check, there is a greater probability of cache misses. This test was run with 10,000 boids because the FPS is zero for simulations with 50,000 boids and greater.   