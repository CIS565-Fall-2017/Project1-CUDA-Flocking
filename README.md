Project 1 CUDA Flocking
====================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

* YAOYI BAI (pennkey: byaoyi)
* Tested on: Windows 10, i7-6700HQ  @2.60GHz 16GB, GTX 980M 8253MB (My own Dell Alienware R3)

### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)


# 1.Naive Boid Simulation #
## Kernel.cu Structure

The structure of the naive sorting would be quite simple. The work flow of all the functions inside Kernel.cu is listed here.


![](https://i.imgur.com/eEBgLtx.jpg)

## Results ##
At first, there are few boids that have velocity, because they are enough close to some other boids. 

***Boids = 5000***

![](https://i.imgur.com/aFRIjlS.jpg)

As time goes by, more and more boids gained velocity and began to move inside the space.

![](https://i.imgur.com/CiOQrMc.jpg)

Boids began to gather together in groups. 

![](https://i.imgur.com/DSgmihS.jpg)

We can also rotate the camera to see different view points.

![](https://i.imgur.com/IKFC3u9.jpg)

***Boids = 10000***

![](https://i.imgur.com/Djef0me.jpg)


# 2. Better Flocking #

## Structure ##

Since the structure is super complicated, I think a graph would be a better illustrator:

![](https://i.imgur.com/yvtWad5.jpg)

The gridIndex3Dto1D and newUpdateVelocity are device procedures that are called from kernComputerIndices and also kernUpdateNeighborSearchScatter. 

## Results ##

***5000 Boids***

![](https://i.imgur.com/cWwECwF.jpg)

![](https://i.imgur.com/xFb5WX4.jpg)

***50000 Boids***

![](https://i.imgur.com/VQsgAqr.jpg)

# 3. Cutting out the middleman #

## Structure ##
 
The structure of Coherent searching are basically the same as the scatter searching. But the difference is that we need a procedure that sort the data of dev_pos and also dev_vel1 according to the index of devBoidIndex. Then we can use the rearrangeed date to calculate the velocity and position update. Finally, copy the memory of dev_vel2 to dev_vel1, and rearranged position value to dev_pos.


# 4. Performance Analysis #

*All the performance analysis are based on Release mode*

5000 Boids

![](https://i.imgur.com/DKviHsM.jpg)

When the total number of boids are 5000, all the fps data of visualized flocking are basically arond 60 fps. Apparently, coherent searching would be more stable and also it reaches the maximum of screen fps before the other two methods. 

However, when the boids are not visualized. The fps will greatly be increased. Apparently, the fps of Naive tends to be stably above 100 fps. And scatter and coherent searching would be around 300 to 400 fps. But I am not sure why the coherent searching is slower than scatter at the beginning. It should be faster to reach the maximum fps than scatter.

50000 Boids

![](https://i.imgur.com/lT69fJI.jpg)

For 50000 boids, it is obvious that naive searching would be too slow because of searching method. The fps of both naive visualized and non-visualized are around 3 fps. 

However, it seems that coherent searching is somehow slower than scatter searching. In my opinion, it is because of the way I implemented this method. Since basically the are the same, but I used two more array in the device, and there will be two more cudaMalloc and cudaFree and also one more cuda kernel function in the main simulation loop, also there is one more cudaMemcpy to copy the rearranged position back to dev_pos. Therefore, it takes more time to transfer data. 

Additionally, I am not sure why there is a severe drop of fps soon after 5 seconds. 

Block Size 512, 5000 boids

![](https://i.imgur.com/I0T99UV.jpg)

When comparing the results of block size = 128 and also block size = 512, I did not see any significant differences. So I think the procedures that do not exceed the maximum storage size of each SM or SP on the GPU, it should not make great differences. But I think there should be a best block size that can make use of all the shared and registers and also global memories on the GPU and attains the best performance when we slightly increase the block size. However, if the block sizes are chosen to be the multiple of 32, which is the warp size for almost all the GPUs, it would guarantee better performance. 

# 5. Questions #



1. **For each implementation, how does changing the number of boids affect performance? Why do you think this is?**

For naive searching, increasing the number of boids will significantly decrease the performance. However, the decreasing of performance for other two searching methods are not apparent, especially when there is a maximum fps during the visualization.


1. **For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

The maximum of block size would be 1024, but I cut the half of it and change 128 to 512 because I did not want my GPU to break. In my experiment, the performance does not make any significant difference. I think it should be that the block size I was testing did not exceed the maximum storage on GPU. If I used more memory space and more threads per block, I will reach some threshold where performance begin to drop significantly. 

1. **For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

I thought it should be faster. But it turned out that it is not super fast, or sometimes it is slower. I think it may because of the way I implemented it. Since I copy the speed and position according to the boidIndex we have sorted into two other arrays. I think the memory copying and allocation of these two arrays takes extra times. Also, since I will have two more "mallocs" and "frees" plus one more "memcpy" for each loop, I think this would make the coherent searching slower. 

1. **Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?**

I have tested. But I don't think 8 neighboring wins absolutely. Since I have tried my best the improve the performance of 8 neighbors. But we need to many if-elses for every boids we want to check (I tried my best to add three more if -elseif-elses than scatter). If-else sometime will cut the performance half, especially when they are not for checking the brim conditions. Therefore, I did not notice significant changes.