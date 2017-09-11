# **University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**





Tested on: Windows 10, Intel Core i7-7700HQ CPU @ 2.80 GHz, 8GB RAM, NVidia GeForce GTX 1050

 ![Built](https://img.shields.io/appveyor/ci/gruntjs/grunt.svg) ![Issues](https://img.shields.io/github/issues-raw/badges/shields/website.svg) ![CUDA 8.0](https://img.shields.io/badge/CUDA-8.0-green.svg?style=flat)  ![Platform](https://img.shields.io/badge/platform-Desktop-bcbcbc.svg)  ![Developer](https://img.shields.io/badge/Developer-Youssef%20Victor-0f97ff.svg?style=flat)



 
 
- [Features](#features)



  - [Analysis](#analysis)
 


 

  - [Observations](#observations)



  - [Blooper](#blooper)
 

 

____________________________________________________


 
 



Simulation using Coherent Uniform Grid at 10k boids:

![In Action](/images/10k-boids.gif)





### Features



- [x] **Naiive Implementation:** 
In the naiive implementation, nearest neighbour search loops across all other boids. Inefficient.


- [x] **Uniform Grid Implementation:** 
Separate Boids into cells and then do nearest neighbour search on boids located in nearby cells.

- [x] **Coherent Grid Implementation:**
A modification and improvement on the Uniform Grid implementation that cuts out having to go through a separate array to check the 
corret index of the sorted positions. I actually added more here as instead of having two separate arrays for positions and velocities,
I interweaved both of them into one array to have them contiguous in memory and thereby speed up look up time. I think that actually
helped a lot as for most people their default boid simulation (5k) for coherent runs around ~550 FPS on much better GPU's than mine. Whereas
my GPU runs it at ~770 FPS easily.





### Analysis:



This is a table of what my FPS count was for each different boid count

![FPS](/images/fps-table.PNG)




This is how it looks in line graph form


![FPS](/images/fps-graph.PNG)






### Observations:

* For each implementation, how does changing the number of boids affect 
performance? Why do you think this is?

Changing the number of boids decreased the framerate. That is because eventually you need more and more blocks to handle the boids.



* For each implementation, how does changing the block count and block size 
affect performance? Why do you think this is?

The performance slowly tapered off to a point for the coherent (not visible). I think that is because at some point the blocks are capable of handling 
the boids appropriately


* For the coherent uniform grid: did you experience any performance improvements 
with the more coherent uniform grid? 
Was this the outcome you expected? 
Why or why not?

Yes, the coherent grid at 50k boids was more than 3x the framerate of the uniform and 40x the framerate of the naiive implementation.
This was what I expected as I put special care in optimizing the coherent more than I probably should have. (See [features](#features)
  list)


* Did changing cell width and checking 27 vs 8 neighboring cells affect performance? 
Why or why not?

Yes an increase of 80% in cell width caused around a halving of the framerate because there was a lot more neighbour search to check.





### Blooper

So this happened. I call it Neon Cube.



![Neon Cube](/images/blooper-1.gif)



