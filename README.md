**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Pennid: lujiayi Name: Jiahao Liu
* Tested on: (TODO) Windows 10, i7-3920XM CPU @ 2.90GHz 3.10 GHz 16GB, GTX 980m SLI 8192MB (personal computer)

# Render Result with Vsync on

![](images/BOIDS.gif)

# Time count and fps count method

* FPS count

In main.cpp, line 238 to line 239 and line 279 to line 282. Count fps for a range of time and then average them to get the result.

* Running time count.

In main.cpp, line 204 to line 207 and line 223 to line 230. Use Cuda event to get time for a range of time and print averaged running time.

# Chart Showing Without Visualization

![](images/chart.png)

![](images/chart1.png)

Roughly speaking the running time per frame for naive method works with time complexity O(n^2). And the other two is O(n).

![](images/chart2.png)

It is a little bit weird that the coherent performs bad on 5000 boids. Maybe the extra time cost for forming new vel1 and pos is much more obvious when the boids number is low.

![](images/chart3.png)

It seems that change of block size is not so important on running time. I guess if we use video card with lower performance this chart will be a lot different.

# Chart Showing With Visualization

![](images/chart4.png)

FPS has a significant drop with visualization on. It is clear that copy data to VBO needs some time.

# Performance with different gridCellWidth

![](images/chart4.png)

* Test Method:

In kernel.cu line 57. Change value to 1 to test block with count of 27.

* Analysis

With 27 neighbor checking has higher fps then 8 neighbor checking, and with #boids increases, the fps advance becomes more obvious. This is because of the total volume checking actually becomes smaller, which means less boids to check in neighbor even with more grids.