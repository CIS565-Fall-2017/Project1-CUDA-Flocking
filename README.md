**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Josh Lawrence
* Tested on: Windows 10, i7-7700HQ @ 2.8GHz 16GB, GTX 1060 6GB  Personal

### (TODO: Your README)


![](images/boidsresults.png)
![](images/boids50k.gif)


GTX 1060 stats:

For each implementation, how does changing the number of boids affect performance? Why do you think this is?
    - For all implementations increaseing the number of boids decreases FPS expoentially. 
    There is simply more work to do with the same amount of resources, more warps pushing more requests on to the global memory bus.  
    It was interesting to see that naive was more efficent at 1000 boids vs the grid implementations. 
    This is likely because the grid implementations have additonal setup overhead and for low number of boids, most of this execution time is spent on this additonal overhead. 

For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
    -Blocksize and count didn't have a drastic effect on perf but it did flucuate enough to notice. 
    There are many factors at play. Each SM on the GPU has a max thread and max block resource limit, whichever comes first.
    Each SM also has a max number of registers divided evenly amonst the threads. 
    If there are more registers needed for the kernel than it has, it can hurt perf when reg vals are getting evicted for new vals.
    In my blocksize graphs, a block size of 16 is the worst performing and 32 onwards is pretty much the same. This is likely because theres nothing going on in half of the warps that are issued from the size 16 blocks since warp size is 32 and pulled from threads in a block. 


For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
    -I did indeed see significant perf improvements but only when the num of boids was very large(50k+). Perhaps for a low number of boids, memory bandwidth and isn't a bottle neck but as the boids get to 50k+ having coherent memory can save a lot of trips to global memory.

Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?
    - huge dip for 5k and 10k boids on grid scattered, the other two methods had similar perf to the 2x cell width (8 neighbors). Don't really know why it would dip in perf on 5k and 10k
