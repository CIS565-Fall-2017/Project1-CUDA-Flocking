**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Ju Yang
* Tested on: Windows 7, i7-4710MQ @ 2.50GHz 8GB, GTX 870M 6870MB (Hasee Notebook K770E-i7)

###README

1. For each implementation, how does changing the number of boids affect performance? Why do you think this is?
* With 128 blocks, I tested from 1000 to 10000 biods.
* For Naive method, the framerate decreases sharply from 1000 to 5000 boids, while it decreases slowly from 5000 to 10000 biods.
* For Uniform grid, the framrate drops sharply from 1000 to 5000 boids, it also drops rather sharp from 5000 to 1000 boids.
* For Coherent grid, the framrate drops sharply from 1000 to 5000 boids, it drops slowly from 5000 to 1000 boids.
* Increasing boid number means increasing all the buffer length, and when that exceeds the GPU warp size, it will lead to more warps, affecting the performance.

2. For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
* For Naive and Uniform Grid method, adding block size decreases the framerate.
* For Coherent Grid method, adding block size increases the framerate.
* For Uniform method, since IndicesArray is involved, more block count means more readings from shared memory, decreasing the performance.
* For Coherent Grid method, I used thrust::copy and thrust::sort_by_key to arrange the pos and vel1. I think thrust functions are more effective than those I wrote.

3. For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
* Unfortunately, with 5000 bioms and 128 blocks, coherent method decreases the framerate from 670 to 411.
* What I expect is, it should increase the framerate in any conditions.
* I think, 5000 bioms are not enough to balance the shortcome of one more thrust::copy and one more thrust::sort_by_key per dt.
If you look at the framerate in 7000 bioms, you will see that's 425 vs 425, and after that, the coherent method keeps leading.

4. Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?
* Well in fact I tried that before, in Debug mode, it's 25fps vs 35fps.
* I think that's because checking 27 neighbours means 27 "for" loops against 8 loops.
* And since there're a lot of "if" branches inside, it do affect a lot to the performance.
* If possible, I wish I won't have to write any "for" loops anymore. 
