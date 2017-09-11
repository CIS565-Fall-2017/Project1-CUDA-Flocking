**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* (TODO) Anton Khabbaz
* Pennkey: akhabbaz
* Tested on: Windows 10 , i7-6600U @ 2.6 GHz 16 GB, GTX 965M 10000MB (personal Windows SUrface book)

### (TODO: Your README)

// update src/CMakeLists.txt  with '-arch=sm_52' since that is the cuda version supported on my graphics card.
p
I added a line to increase the frame rate by turning off vsync with
glfwSwapInterval(false) (Josh helped me).


Got 1.2 and the flocking to work in the naive case.  I changed parameters
and found that with 1000 boids I got a frame rate of 26 fps with my gpu.   The flocks converge and I find no problems. 

I wrote the code for 2.1 the grid search.  I am pretty sure I understood the grid search and I wrote and compiled the code.
I ran it and the Boids don't move.  

I determined that the gridCells and the start and endIndices are correct.  This is all the input needed to update the velocity.
I am sure that the indices.  I ran the code with 100 boids and a 20 sized max radius.  These parameters worked for the naive grid and had the boids converge.  I used those to get the uniform grid scatter to work.  With those, I wrote a Boid::testGRidArray that transfeerred the intArrays to the host and printed them out nicely.  

Those indicated that the Start, End grid indices were correct.  Those algorithms that I wrote were all parallel, and had no loops over all the Boids.

Next I tried to test the updateVelocityScatter with the same parameters.  This failed because  I couldn't get the debugger to stop at the first line.  I got this message that i posted to the google groups that the application is in break mode and the point can't be hit.  I could hit the break point in the initSimulation Kernel, but not in the updateSim kernel.  I tried also with the kernel from 1.2 and that also did not work. 

I get this error printed out:

CUDA error 73 [c:\program files\nvidia gpu computing toolkit\cuda\v8.0\include\thrust\system\cuda\detail\cub\device\dispatch/device_radix_sort_dispatch.cuh, 735]: an illegal instruction was encountered
CUDA error 73 [c:\program files\nvidia gpu computing toolkit\cuda\v8.0\include\thrust\system\cuda\detail\cub\device\dispatch/device_radix_sort_dispatch.cuh, 752]: an illegal instruction was encountered
CUDA error 73 [c:\program files\nvidia gpu computing toolkit\cuda\v8.0\include\thrust\system\cuda\detail\cub\device\dispatch/device_radix_sort_dispatch.cuh, 875]: an illegal instruction was encountered


Without Cuda debugging, it is pretty hard to make sure those algorithms on the device work.
Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)
