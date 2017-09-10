**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Yash Vardhan
* Tested on: Windows 10 Pro, Intel i5-4200U (4) @ 2.600GHz 4GB, NVIDIA GeForce 840M 2048MB

Boids Flocking using coherent uniform grid
===========================================

![](images/boids.gif)

Performance Analysis
====================

FPS vs Number of Boids(Visualized)
----------------------------------

![](images/visualboid.jpg)

FPS vs Number of Boids(Non-Visualized)
--------------------------------------

![](images/nonvisualboid.jpg)

FPS vs Block Size tested on 20,000 boids
----------------------------------------

![](images/blocksize.jpg)

Q/A
===

**Q1 - For each implementation, how does changing the number of boids affect performance? Why do you think this is?**

A1 - Naive neighbour search has an asymptotic drop with increase in the number of boids. Updating the velocities and position of boids gets moreand more difficult as the number of boid increases. Both scattered and coherent uniform grid flocking algorithms almost have a linear decrease with the number of increasing boids, but uniform grid flocking experiences a steeper gradient inintially which could be explained due to chasing the grid cell pointers. 

Also if the number of boids is 1000 or less, the naive flocking algorithm performs better than the uniform grid algorithms, this could be explained to the sparse grid the algorithms face, costing them unneccessary sorting computations.

**Q2 - For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

A2 - In naive neighbour implementation, increasing the block size barely affects the performance as it resembles a straight line with no slope. On the other hand the 2 grid algorithms show a slight increase in performance over block size of 128 and then come down to resemble a straight line, which helps explain why 128 block size was chosen as the default one for this project. In my opinion this is a purely experimental observation.


**Q3 - For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

A3 - Coherent uniform grid algorithm scales better than its counterparts with large number of boids, where as other 2 algorithms just fail. This could be linked to the lesser time expenditure in keeping up with the pointers(as in uniform grid) or having to compute all those other boid velocities in naive neighbour implementation.


**Q4 - Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?**

A4 - Checking 27 neighbouring cells, in my case gave a slighly higher FPS than checking 8 neighbouring cells on larger number of boids. This might be explained through the grid cell width, which is twice the radius in 8 cell check case, so greater number of boids need to be covered.