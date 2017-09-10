**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Yuxin Hu
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 8GB, GTX 960M 4096MB (Personal Laptop)

### README
#### Results
![Naive Simulation 5000 Particles 27 Neighbor Search](/images/Naive5000neighbor27.gif)
![Uniform Simulation 5000 Particles 27 Neighbor Search](/images/Uniform5000neighbor27.gif)
![Coherent Simulation 5000 Particles 27 Neighbor Search](/images/Coherent5000neighbor27.gif)
![Coherent Simulation 50000 Particles 27 Neighbor Search](/images/Coherent50000neighbor27.gif)
![Coherent Simulation 500000 Particles 27 Neighbor Search](/images/Coherent500000neighbor27.gif)

####Performance Analysis
1. Performance Comparason between 3 simulation methods
![3 Simulations Performance Comparason With Visual](/images/PerformAnalysis3MethodsVisual.PNG)
![3 Simulations Performance Comparason Without Visual](/images/PerfomAnalysis3MethodsNoVisual.PNG)

Uniform Grids and Coherent Grids Search performs significantly better than Naive Simulation, because the previous two simulations reduce the number of boids need to be checked drastically. The coherent grid search performs a little bit better than uniform grid search without Visual. With Visual simulation they two perform almost the same. Coherent grid is supposed to perform a little bit better because it reduces one level of data retrieval from global memory, although the cost is we need extra space to store the shuffled data.

2. Performance Analysis as boids number increases.
![Naive Simulation With Visual](/images/PerformAnalysisNaiveParticleNumberChangeVisual.PNG)
![Naive Simulation Without Visual](/images/PerformAnalysisNaiveParticleNumberChangeNoVisual.PNG)
![Uniform Simulation With Visual](/images/PerformAnalysisUniformParticleNumberChangeVisual.PNG)
![Uniform Simulation Without Visual](images/PerformAnalysisUniformParticleNumberChangeNoVisual.PNG)
![Coherent Simulation With Visual](/images/PerformAnalysisCoherentParticleNumberChangeVisual.PNG)
![Coherent Simulation Without Visual](/images/PerformAnalysisCoherentParticleNumberChangeNoVisual.PNG)

From the plots we can observe that as the boids number increases, the performance drops, with and without visual, in all three simulations. Different simulation has different limit on how many boids can simulate. Clearly coherent can support most number of boids.

3. Performance Analysis as block size changes.
![Coherent Simulation with 50000 Boid Particles, with visual](/images/PerformAnalysisBlockSizeChange.PNG)

From the tests I did, I observed that as the block size decreases, the performance improves.

4. Other Observations.
![Coherent Simulation 50000 Boids With Visual Neighbor Search Comparason](/images/PerformAnalysisNeighborSearch.PNG)
![Coherent Simulation 50000 Boids With Visual Neighbor Search Comparason After Improvement](/images/PerformAnalysisNeighborSearchImproved.PNG)

The first plot is a comparason between the same simulation using different neighbor search. One searches 8 neighbor cells with each cell width 2 times the distance. The other searches 27 neighbor cells with each cell width the same as the distance. In this implementation, when I calculated the cellStartIndex and cellEndIndex, I did not use a parallel unlooping methos. I simply loop over the particleGridIndices array, check is there is an index identical to the current cell Index. Although there is some optimization to prevent it from going through all particleGridIndices, but the worst case is still O(n). As the total number of cells increase, this brute force search will significally drag down the performance. After I improved this method, the complexity for each thread is O(1), and when searching over neighboring 27 grid cells, the total volume is 27*distance^3, which is smaller than volume 8*8*distance^3, therefore the total number of boids need to be checked actually decreases. The performance comparason in second plot above shows the opposite effect after this improvement.

