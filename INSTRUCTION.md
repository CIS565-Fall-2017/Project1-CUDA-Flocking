CUDA Introduction - Flocking
============================

This is due Sunday, September 10.

**Summary:** In this project, you will get some real experience writing simple
CUDA kernels, using them, and analyzing their performance. You'll implement a
flocking simulation based on the Reynolds Boids algorithm, along with two levels
of optimization: a uniform grid, and a uniform grid with semi-coherent memory access.

## Part 0: Nothing New

This project (and all other CUDA projects in this course) requires an NVIDIA
graphics card with CUDA capability. Any card with Compute Capability 2.0
(`sm_20`) or greater will work. Check your GPU on this
[compatibility table](https://developer.nvidia.com/cuda-gpus).
If you do not have a personal machine with these specs, you may use those
computers in the Moore 100B/C which have supported GPUs, but pay close
attention to the **setup** section below.

**HOWEVER**: If you need to use the lab computer for your development, you will
not presently be able to do GPU performance profiling. This will be very
important for debugging performance bottlenecks in your program. If you do not
have administrative access to any CUDA-capable machine, please email the TA.

## Part 1: Naive Boids Simulation

### 1.0. Setup - The Usual

See Project 0, Parts 1-3 for reference.

If you are using the Nsight IDE (not Visual Studio) and started Project 0
early, note that things have
changed slightly. Instead of creating a new project, use
*File->Import->General->Existing Projects Into Workspace*, and select the
project folder as the root directory. Under *Project->Build
Configurations->Set Active...*, you can now select various Release and Debug
builds.

* `src/` contains the source code.
* `external/` contains the binaries and headers for GLEW, GLFW, and GLM.

**CMake note:** Do not change any build settings or add any files to your
project directly (in Visual Studio, Nsight, etc.) Instead, edit the
`src/CMakeLists.txt` file. Any files you create must be added here. If you edit
it, just rebuild your VS/Nsight project to sync the changes into the IDE.

Please run the project without modifications to ensure that everything works
correctly. We have provided a space in `kernel.cu` to run test code, and at the
moment it contains an example of how to use the `Thrust` library to perform
key-value sorting on the GPU. If everything is working right, you should see
some output in the console window and a cube of gray particles. The viewer is
equipped with camera controls: left-click and drag to move the camera view,
right-click and drag vertically to zoom in and out.

On some configurations (commonly on lab machines), the unmodified code built in
`release` mode crashes mysteriously with an
`exception: thrust::system::system_error` at some memory location.
This can happen because the compute capability specified in `src/CMakeLists.txt`
does not match your GPU's compute capability.
Look up your compute capability version and change the `-arch=sm_20` flag to
match. `sm_20` corresponds with compute capability 2.0, `sm_21` with 2.1,
`sm_50` with 5.0, and so on.

You will need to re-generate the project with cmake.

### 1.1. Boids with Naive Neighbor Search

In the Boids flocking simulation, particles representing birds or fish
(boids) move around the simulation space according to three rules:

1. cohesion - boids move towards the perceived center of mass of their neighbors
2. separation - boids avoid getting to close to their neighbors
3. alignment - boids generally try to move with the same direction and speed as
their neighbors

These three rules specify a boid's velocity change in a timestep.
At every timestep, a boid thus has to look at each of its neighboring boids
and compute the velocity change contribution from each of the three rules.
Thus, a bare-bones boids implementation has each boid check every other boid in
the simulation.

Here is some quick pseudocode to help you out:

#### Rule 1: Boids try to fly towards the centre of mass of neighbouring boids

```
function rule1(Boid boid)

    Vector perceived_center

    foreach Boid b:
        if b != boid and distance(b, boid) < rule1Distance then
            perceived_center += b.position
        endif
    end

    perceived_center /= N-1

    return (perceived_center - boid.position) * rule1Scale
end
```

#### Rule 2: Boids try to keep a small distance away from other objects (including other boids).

```
function rule2(Boid boid)

    Vector c = 0

    foreach Boid b
        if b != boid and distance(b, boid) < rule2Distance then
            c -= (b.position - boid.position)
        endif
    end

    return c * rule2Scale
end
```

#### Rule 3: Boids try to match velocity with near boids.

```
function rule3(Boid boid)

    Vector perceived_velocity

    foreach Boid b
        if b != boid and distance(b, boid) < rule3Distance then
            perceived_velocity += b.velocity
        endif
    end

    perceived_velocity /= N-1

    return perceived_velocity * rule3Scale
end
```
Based on [Conard Parker's notes](http://www.vergenet.net/~conrad/boids/pseudocode.html) with slight adaptations. For the purposes of an interesting simulation,
we will say that two boids only influence each other according if they are
within a certain **neighborhood distance** of each other.

We also have a simple [2D implementation in Processing](http://studio.sketchpad.cc/sp/pad/view/ro.9cbgCRcgbPOI6/rev.23)
that is very conceptually similar to what you will be writing. Feel free to use
this implementation as a math/code reference.

For an idea of how the simulation "should" look in 3D,
[here's what our reference implementation looks like.](https://vimeo.com/181547860)

### 1.2. Code walkthrough

* `src/main.cpp`: Performs all of the CUDA/OpenGL setup and OpenGL
  visualization.
* `src/kernel.cu`: CUDA device functions, state, kernels, and CPU functions for
  kernel invocations. In place of a unit testing/sandbox framework, there is
  space in here for individually running your kernels and getting the output
  back from the GPU before running the actual simulation. PLEASE make use of
  this in Part 2 to individually test your kernels.

1. Search the code for `TODO-1.2` and `LOOK-1.2`.
   * `src/kernel.cu`: Use what you learned in the first lectures to
     figure out how to resolve these X Part 1 TODOs.

### 1.3. Play around

The parameters we have provided result in a stable simulation using our
reference implementation, but your mileage may vary. Play around with the boid
count as well and see how the simulation responds.

## Part 2: Let there be (better) flocking!

### 2.0. A quick explanation of uniform grids

Recall in part 1 that any two boids can only influence each other if they are
within some *neighborhood distance* of each other.
Based on this observation, we can see that having each boid check every
other boid is very inefficient, especially if (as in our standard parameters)
the number of boids is large and the neighborhood distance is much smaller than
the full simulation space. We can cull a lot of neighbor checks using a
datastructure called a **uniform spatial grid**.

A uniform grid is made up of cells that are at least as wide as the neighborhood
distance and covers the entire simulation domain.
Before computing the new velocities of the boids, we "bin" them into the grid in
a preprocess step.
![a uniform grid in 2D](images/Boids%20Ugrid%20base.png)

If the cell width is double the neighborhood distance, each boid only has to be
checked against other boids in 8 cells, or 4 in the 2D case.

![a uniform grid in 2D with neighborhood and cells to search for some particles shown](images/Boids%20Ugrid%20neighbor%20search%20shown.png)

You can build a uniform grid on the CPU by iterating over the boids, figuring out
its enclosing cell, and then keeping a pointer to the boid in a resizeable
array representing the cell. However, this doesn't transfer well to the GPU
because:

1. We don't have resizeable arrays on the GPU
2. Naively parallelizing the iteration may lead to race conditions, where two
particles need to be written into the same bucket on the same clock cycle.

Instead, we will construct the uniform grid by sorting. If we label each boid
with an index representing its enclosing cell and then sort the list of
boids by these indices, we can ensure that pointers to boids in the same cells
are contiguous in memory.

Then, we can walk over the array of sorted uniform grid indices and look at
every pair of values. If the values differ, we know that we are at the border
of the representation of two different cells. Storing these locations in a table
with an entry for each cell gives us a complete representation of the uniform
grid. This "table" can just be an array with as much space as there are cells.
This process is data parallel and can be naively parallelized.
![buffers for generating a uniform grid using index sort](images/Boids%20Ugrids%20buffers%20naive.png)

### 2.1. Code walkthrough

Instead of having you implement parallel sorting on the GPU for your first
homework, we will use the value/key sort built into **Thrust**. See
`Boids::unitTest` in `kernel.cu` for an example of how to use this.

Your uniform grid will probably look something like this in GPU memory:
- `dev_particleArrayIndices` - buffer containing a pointer for each boid to its
data in dev_pos and dev_vel1 and dev_vel2
- `dev_particleGridIndices` - buffer containing the grid index of each boid
- `dev_gridCellStartIndices` - buffer containing a pointer for each cell to the
beginning of its data in `dev_particleArrayIndices`
- `dev_gridCellEndIndices` - buffer containing a pointer for each cell to the
end of its data in `dev_particleArrayIndices`.

Here the term `pointer` when used with buffers is largely interchangeable with
the term `index`, however, you will effectively be using array indices as
pointers.

See the TODOs and LOOKs for Part 2.1 in the code for some pseudocode help.

You can toggle between different timestep update modes using the defines in
`main.cpp`.

### 2.2 Play around some more
Compare your uniform grid velocity update to your naive velocity update.
In the typical case, the uniform grid version should be considerably faster.
Try to push the limits of how many boids you can simulate.

Change the cell width of the uniform grid to be the neighborhood distance, instead of twice the neighborhood distance. Now, 27 neighboring cells will need to be checked for intersection. Does this increase or decrease the efficiency of the flocking?

### 2.3 Cutting out the middleman
Consider the uniform grid neighbor search outlined in 2.1: pointers to boids in
a single cell are contiguous in memory, but the boid data itself (velocities and
positions) is scattered all over the place. Try rearranging the boid data
itself so that all the velocities and positions of boids in one cell are also
contiguous in memory, so this data can be accessed directly using
`dev_gridCellStartIndices` and `dev_gridCellEndIndices` without
`dev_particleArrayIndices`.

![buffers for generating a uniform grid using index sort, then making the boid data coherent](images/Boids%20Ugrids%20buffers%20data%20coherent.png)

See the TODOs for Part 2.3. This should involve a slightly modified copy of
your code from 2.1.

## Part 3: Performance Analysis
For this project, we will guide you through your performance analysis with some
basic questions. In the future, you will guide your own performance analysis -
but these simple questions will always be critical to answer. In general, we
want you to go above and beyond the suggested performance investigations and
explore how different aspects of your code impact performance as a whole.

The provided framerate meter (in the window title) will be a useful base
metric, but adding your own `cudaTimer`s, etc., will allow you to do more
fine-grained benchmarking of various parts of your code.

REMEMBER:
* Do your performance testing in Release mode!
* Performance should always be measured relative to some baseline when
  possible. A GPU can make your program faster - but by how much?
* If a change impacts performance, show a comparison. Describe your changes.
* Describe the methodology you are using to benchmark.
* Performance plots are a good thing.

### Questions

There are two ways to measure performance:
* Disable visualization so that the framerate reported will be for the the
  simulation only, and not be limited to 60 fps. This way, the framerate
  reported in the window title will be useful.
  * To do this, change `#define VISUALIZE` to `0`.
* For tighter timing measurement, you can use CUDA events to measure just the
  simulation CUDA kernel. Info on this can be found online easily. You will
  probably have to average over several simulation steps, similar to the way
  FPS is currently calculated.

This section will not be graded for correctness, but please let us know your
hypotheses and insights.

**Answer these:**

* For each implementation, how does changing the number of boids affect
performance? Why do you think this is?
* For each implementation, how does changing the block count and block size
affect performance? Why do you think this is?
* For the coherent uniform grid: did you experience any performance improvements
with the more coherent uniform grid? Was this the outcome you expected?
Why or why not?
* Did changing cell width and checking 27 vs 8 neighboring cells affect performance?
Why or why not?

**NOTE: Nsight performance analysis tools *cannot* presently be used on the lab
computers, as they require administrative access.** If you do not have access
to a CUDA-capable computer, the lab computers still allow you to do timing
mesasurements! However, the tools are very useful for performance debugging.


## Part 4: Write-up

1. Take a screenshot of the boids **and** use a gif tool like [licecap](http://www.cockos.com/licecap/) to record an animations of the boids with a fixed camera.
Put this at the top of your README.md. Take a look at [How to make an attractive
GitHub repo](https://github.com/pjcozzi/Articles/blob/master/CIS565/GitHubRepo/README.md).
2. Add your performance analysis. Graphs to include:
- Framerate change with increasing # of boids for naive, scattered uniform grid, and coherent uniform grid (with and without visualization)
- Framerate change with increasing block size

## Submit

If you have modified any of the `CMakeLists.txt` files at all (aside from the
list of `SOURCE_FILES`), mention it explicitly. Beware of any build issues discussed on the Google Group.

Open a GitHub pull request so that we can see that you have finished.
The title should be "Project 1: YOUR NAME".
The template of the comment section of your pull request is attached below, you can do some copy and paste:  

* [Repo Link](https://link-to-your-repo)
* (Briefly) Mentions features that you've completed. Especially those bells and whistles you want to highlight
    * Feature 0
    * Feature 1
    * ...
* Feedback on the project itself, if any.


And you're done!

## Tips
- If your simulation crashes before launch, use
`checkCUDAErrorWithLine("message")` after CUDA invocations
- `ctrl + f5` in Visual Studio will launch the program but won't let the window
close if the program crashes. This way you can see any `checkCUDAErrorWithLine`
output.
- For debugging purposes, you can transfer data to and from the GPU.
See `Boids::unitTest` in `kernel.cu` for an example of how to use this.
- For high DPI displays like 4K monitors or the Macbook Pro with Retina Display, you might want to double the rendering resolution and point size. See `main.hpp`.
- Your README.md will be done in github markdown. You can find a [cheatsheet here](https://guides.github.com/pdfs/markdown-cheatsheet-online.pdf). There is
also a [live preview plugin](https://atom.io/packages/markdown-preview) for the
[atom text editor](https://atom.io/) from github. The same for [VS Code](https://www.visualstudio.com/en-us/products/code-vs.aspx)
- If your framerate is capped at 60fps, [disable V-sync](http://support.enmasse.com/tera/enable-v-sync-to-fix-graphics-issues-screen-tearing)

## Optional Extra Credit
Add fast nearest neighbor search using shared memory and the uniform grid. If you choose to tackle this problem, have it done before Project 5. Include additional graphs and performance analysis, showing clearly how much better the program performed using shared memory.
