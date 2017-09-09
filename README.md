**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Pennid: lujiayi Name: Jiahao Liu
* Tested on: (TODO) Windows 10, i7-3920XM CPU @ 2.90GHz 3.10 GHz 16GB, GTX 980m SLI 8192MB (personal computer)

# Render Result with Vsync on

![](images/BOIDS.gif)


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