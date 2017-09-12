**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Xincheng Zhang
* Tested on: Windows 10, i7-4702HQ @ 2.20GHz 8GB, GTX 870M 3072MB (Personal Laptop)



## ScreenShot

___


![](https://github.com/XinCastle/Project1-CUDA-Flocking/blob/master/images/2017.09.11.23.34.16.gif)


*This result is tested under the following settings:
number of particles: 5000
blocksize: 128
coherent-uniform
rule1Distance 5.0f
rule2Distance 3.0f
rule3Distance 5.0f
rule1Scale 0.01f
rule2Scale 0.1f
maxSpeed 1.0f


## Performance Analysis

___

The performance of **Naive Boids**, **Uniform Boids** and **Coherent Uniform Boids** is measured by FPS.
The following is the diagram comparing these three methods.

![](https://github.com/XinCastle/Project1-CUDA-Flocking/blob/master/images/diagram.png)

**Note:I find the movement of mine looks the same as the example. However, the frame rates of uniform boids
 and coherent uniform boids are much slower than I expect. I checked my algorithm but wasn't able to tell why
 it happens.

 In conclusion, when the boids number is more than 1000, the performance order is: Coherent Uniform Boids > Uniform Boids > Naive Boids


### Questions

___
*For each implementation, how does changing the number of boids affect performance? Why do you think this is?

1. As the number of boids increases, the performance becomes worse for all the three methods. As for how much they are affected, the order
is: Naive Boids > Uniform Boids ~=(almost equal to) Cohereent Uniform Boids. I think the reason is because as we add more boids in the system,
there are more neighbors around a particle so that the whole algorithm requires more time for calculation. The naive approach is the least
efficient, so it decreases faster than the other two methods.


*For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

2. As the blocksize becomes larger, the frame rate slightly drops. I think the reason is the following:
It's less efficient to fetch and access data in larger memory. Therefore, as the blocksize becomes larger, it takes more and more time for
the system to access the memory.


*For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

3. Yes, the coherent uniform grid has better performance than the uniform grid because reading data which is stored in contiguous memory is faster.
Therefore, the main reason of the improvement is because of contiguous memory.


*Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?

4. Yes, when the number of neighboring cells becomes 27 rather than 8, the performance becomes better. I think it's because as cell width decreases,
the total particles defined as "neighboring particles" significantly decreases. Therfore, the time cost decreases although we have more cells to loop
in the function.