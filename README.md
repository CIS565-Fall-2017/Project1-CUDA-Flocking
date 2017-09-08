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

## Result ##
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

## Performance Analysis ##

# 2. Better Flocking #

## Structure ##

Since the structure is super complicated, I think a graph would be a better illustrator:

![](https://i.imgur.com/yvtWad5.jpg)

## Result ##

26 Neighbors Searching 

![](https://i.imgur.com/cWwECwF.jpg)

# 3. Cutting out the middleman #

## Structure ##
 
The structure of Coherent searching are basically the same as the scatter searching. But the difference is that we need a procedure that sort the data of dev_pos and also dev_vel1 according to the index of devBoidIndex. Then we can use the rearrangeed date to calculate the velocity and position update. Finally, copy the memory of dev_vel2 to dev_vel1, and rearranged position value to dev_pos.


# 4. Performance Analysis #

*All the performance analysis are based on Release mode*

5000 Boids

![](https://i.imgur.com/DKviHsM.jpg)

50000 Boids

![](https://i.imgur.com/lT69fJI.jpg)

Block Size 512, 5000 boids

![](https://i.imgur.com/I0T99UV.jpg)

