Project 1 Boids with CUDA
==========================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Boids**

* Aman Sachan
* Tested on: Windows 10, i7-7700HQ @ 2.8GHz 32GB, GTX 1070(laptop GPU) 8074MB (Personal Machine: Customized MSI GT62VR 7RE)

### Overview

![](images/results/Boids.gif)
###### (Run on GTX 1070(laptop GPU) with 10000 boids)

This project served as an introduction to CUDA kernels, how to use them, and how to analyze their performance. This was done by implementing the Reynolds Boids algorithm in parallel on the GPU.

Boids is a crowd simulation algorithm developed by Craig Reynolds in 1986 which is modeled after the flocking behaviors exibited by birds and fish. The most basic version of the algorithm operates on three rules:
* Cohesion: Boids will try to move towards the perceived center of mass of other boids around them.
* Alignment: Boids will tend to steer in the direction of the perceived average movement of other boids around them.
* Separation: Boids will try to keep some amount of distance between them and other boids.

### Three implementations:
#### 1. Naive

The naive approach for computing the new positions and velocities for the boids is to check each boid against every other boid and apply the three rules described above. This is extremely slow with a time complexity of O(N^2). 

A simple way to implement this in CUDA is to have one thread for each boid. Each thread would loop over the entire position and velocity buffers (skipping itself) and compute a new velocity for that boid. Then, another CUDA kernel would apply the position update using the new velocity data.

#### 2. Uniform Scattered Grid

It is quite obvious that a spatial data structure can greately improve performance for the algorithm. If we reduce the number of boids each boid needs to check against, we can decrease the amount of work each thread needs to do. Because the three rules only apply within a certain radius, organizing the boids on a uniformly spaced grid allows us to perform a very efficient neighbor search and only check a very limited number of boids. We choose our uniform grid cell width to be twice the maximum search radius. This makes it so that at most, a boid may be influenced by other boids in the 8 cells directly around it.

To create the "grid" we create an additional CUDA kernel which fills a buffer with respective grid cell indices. A parallel sorting algorithm, thrust::sort_by_key, is then used to sort boid indices by their corresponding grid cell indices. Furthermore, we store the "start" and "end" indicies for each grid cell in two new buffers by checking for transitions in the sorted grid cell buffer.

Now, instead of checking against all other boids, each thread in our velocity update can determine the boids grid index based on its position; this then allows us to determine the 8 neighboring cells in 3D space, and only apply rules for those boids that have an index between "start" and "end" of a neighboring cell.

#### 3. Uniform Coherent Grid

We can make the uniform grid spatial structure better by making memory accesses more coherent. The uniform scattered grid implementation does unecessary hopping and "pointer"-chasing. Instead of using an additional sorted buffer to find the index into our particle data, we can shuffle the particle data so that it is also sorted by grid cell index. This allows the GPU to load in and cache our position and velocity data for the boids resulting in fewer global memory calls. This small change ends up making signficant performance improvements.

### Performance Analysis

#### Effect of boid count on framerate

Tests were done using a block size of 128

![](images/results/BoidsVSframerateData.png)

![](images/results/NumberofBoidsVSframerate_LineChart.png)

##### DataPoints for Naive are missing beyond 100000 boids, and Uniform scattered grid beyond 1 Million as the CUDA kernels refused to launch with such high boid counts for those two approaches.

As would be expected, as the number of boids increases, all 3 approaches suffer slowdowns. However, the line graph above clearly shows drastic things. We can see that using the Uniform Scattered Grid greatly improves performance and then the Uniform Coherent Grid surpasses even that in terms of performance. Intuitively, this makes sense. Using the Uniform Scattered Grid we greatly reduced the number of boids that have to be checked against for each boid. Making the boid data more memory coherent allowed us to access data(memory) faster because of better caching and reduction in the number of calls to global memory. This made Uniform Coherent Grids Perform that much better. 

![](images/results/NumberofBoidsVSframerate_BarChart.png)

If we zoom into to the first few data points, we can notice a sudden jump that almost doubles our framerate while increasing the number of boids. At first glance, this seems kind of absurd, but it is more likely that in some situations, the number of boids do not map well into the memory of the underlying architecture which leads to a frustrating reduction in framerate.

#### Effect of block size on framerate

![](images/results/BlocksizeVSframerateData.png)

![](images/results/BlocksizeVSframerate.png)

#### Drastic fall in framerate

![](images/results/NumberofBoidsVSframerate_LineChart_Weirdness.png)

Note the large jumps at about (5300, 5500), (16400, 16500), (31200, 31300), and (43600, 43700). I think that this may be happening because in some situations, the number of boids does not map well to the underlying architecture and memory access becomes less efficient.

### Worst-Case Scenario for Uniform and Coherent Grids









Trying to understand these results are likely specific to the architecture or the used GPU, so I will try to take this into accound in my explanation. The compute capability of my GPU (GT750M) is 3.0.

**Grid dimensions**: For my GPU, I have a maximum of _2^(31) - 1_ blocks per grid in the x-dimension. Therefore, we likely don't ever hit a bottle neck in the number of blocks we can launch given that even at 16 threads per block, we only need 128 blocks.

**Blocks per SM**: The GT750M can handle up to 16 blocks per multiprocessor. This means that, in the case of using 16 threads per block, we can only hold 16 of the 128 blocks in a single SM at a time.The GT750M also only has 2 multiprocessors, which means that 32 of the 128 blocks can be run at a time. This means that the GPU has to wait before it can load the rest of the blocks in, leading to a slower simulation. Once we raise the number of threads per block to 64, we create 32 blocks out of the 2048 boid threads. This allows us to launch all the threads at once on the GPU. This makes sense with our scattered and coherent results since there is a sharp framerate increase from a block size of 16 until 64, after which the increase begins to taper off.

**Registers per block**: The GT750M has a maximum of 65536 registers per block. This might explain why the rate at which our framerate increases starts to fall off after hitting 64 threads per block. As we add more threads per block, we are expecting each block to handle more memory since we are adding more boid states to handle. This means that at 1024 threads per block, each boid only has 64 registers to use. If we exceed this, data in registers likely need to be moved to and from global memory, leading to an additional overhead in memory management.

**Threads per SM**: The GT750M also only handles up to 2048 threads per SM. Considering the maximum number of threads per block we can use is 1024 and we only have 2048 boids in our simulation, we can easily run the simulation with two blocks in a single multiprocessor. This means that as the number of threads per block increases to 1024, the simulation speed will increase but not take a penalty from here.

If anyone reaches this far and happens to see any issues with my analysis or something I missed, please feel free to submit an issue and let me know!