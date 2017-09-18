#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"





// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128 //Orign=128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.乒乓缓存？
//(We need 2 velocity buffers in parral computing)
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;



int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

int* dev_origin;//Use this to record the grid_index we got.


// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");
  //2.3
  cudaMalloc((void**)&dev_origin, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  //
  cudaThreadSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  glm::vec3 thisPos;

  if (index < N) {

	thisPos= pos[index];

    vbo[4 * index + 0] = thisPos.x * c_scale;
    vbo[4 * index + 1] = thisPos.y * c_scale;
    vbo[4 * index + 2] = thisPos.z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaThreadSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids

	glm::vec3 center;//群中心, center of mass
	glm::vec3 push;//斥力, pushing force
	glm::vec3 speed;//群速度, velocity of all neighbours

	glm::vec3 range;//range of affection
	int rule_count[3];//How many bios are included in each rule

	//init
	center = glm::vec3(0);
	push = glm::vec3(0);
	speed = glm::vec3(0);
	rule_count[0] = rule_count[1] = rule_count[2] = 0;
	//
	int i;
	float length;

	for (i = 0; i < N; i++)
	{
		if (i != iSelf)//don't count itself
		{
			range = pos[i] - pos[iSelf];
			length = sqrt(range.x*range.x + range.y*range.y + range.z*range.z);

			if (length < rule1Distance)// centralizing effect
			{
				center = center + range;
				rule_count[0]++;
			}
			if (length < rule2Distance)// pushing each other if too near
			{
				push = push - range;
				rule_count[1]++;
			}
			if (length < rule3Distance)//follow neighbour's speed
			{
				speed = speed + vel[i];
				rule_count[2]++;
			}
		}
	}//end of loop

	//Compute Averange here
	if (rule_count[0] != 0)
	{
		center = center * (1 / (float)rule_count[0]);
	}
	if (rule_count[1] != 0)
	{
		push = push * (1 / (float)rule_count[1]);
	}
	if (rule_count[2] != 0)
	{
		speed = speed * (1 / (float)rule_count[2]);
	}
	//local vars
	glm::vec3 out;
	float out_len;
	//Compute new velocity and its speed
	out = vel[iSelf] + center*rule1Scale + push*rule2Scale + speed*rule3Scale;
	out_len = sqrt(out.x*out.x + out.y*out.y + out.z*out.z);
	//Normalize the speed
	if (out_len > maxSpeed)
	{
		out = out*(maxSpeed / out_len);
	}
	return out;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
  //因为vel1后面咱还得用呢!(Because we need vel1 remain unchanged while parral computing velocity)

	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < N)
	{
		vel2[index] = computeVelocityChange(N, index, pos, vel1);
	}
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them

  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order? 
//(It depends on the shape of simulation space. We start from the longest edge. But since it's square, anyway...)
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	glm::vec3 new_pos;
	int num;

	glm::vec3 a(0.5f);
	a *= gridResolution;

	if (index < N)
	{
		new_pos = pos[index]*inverseCellWidth + a;
		num=gridIndex3Dto1D(new_pos.x, new_pos.y, new_pos.z, gridResolution);

		if (num < gridResolution*gridResolution*gridResolution)//Double Check here
		{
			indices[index] = index;//write array for arrange
			gridIndices[index] = num;//write to grid index
		}
	}
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices, int *particleArrayIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int grid;
	if (index > 0)
	{
		if (particleGridIndices[index - 1] != particleGridIndices[index])
		{
			grid = particleGridIndices[index];//Which grid is it in?

			if (grid > 0)
			{
				gridCellEndIndices[grid -1] = index - 1;
				gridCellStartIndices[grid] = index;
			}
		}
	}
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	float range= imax(imax(rule1Distance, rule2Distance), rule3Distance);
	glm::vec3 neighbor;
	int neighbor_num;
	float rate = range / cellWidth;	
	glm::vec3 new_pos; 
	glm::vec3 a(0.5f);
	a *= gridResolution;
	new_pos = pos[index] * inverseCellWidth + a;
	int pos_grid = gridIndex3Dto1D(new_pos.x, new_pos.y, new_pos.z, gridResolution);
	int i,j,k;
	float length;
	glm::vec3 len;
	float distance;
	int start, end;
	glm::vec3 member;

	glm::vec3 Vertex[8];//The eight Vertexs of the grid
	Vertex[0] = glm::vec3(0, 0, 0);
	Vertex[1] = glm::vec3(0, 0, 1);
	Vertex[2] = glm::vec3(0, 1, 0);
	Vertex[3] = glm::vec3(0, 1, 1);
	Vertex[4] = glm::vec3(1, 0, 0);
	Vertex[5] = glm::vec3(1, 0, 1);
	Vertex[6] = glm::vec3(1, 1, 0);
	Vertex[7] = glm::vec3(1, 1, 1);

	////////////////////////////
	glm::vec3 center;// center of mass
	glm::vec3 push;// pushing force
	glm::vec3 speed;//velocity of all neighbours
	int rule_count[3];//How many bios are included in each rule
	center = glm::vec3(0);
	push = glm::vec3(0);
	speed = glm::vec3(0);
	rule_count[0] = rule_count[1] = rule_count[2] = 0;
	////////////////////////////////

	int new_neighbour[8];// Possible Grid Index of the neighbouring grids

	for (i = 0; i < 8; i++)// See if this grid is within affection range
	{
		len = new_pos + Vertex[i];//Check each vertex
		distance = sqrt(len.x*len.x + len.y*len.y + len.z*len.z);
		new_neighbour[i] = -1;//Init as -1

		if (distance < range)//if vertex is within range
		{
			neighbor = new_pos + Vertex[i];

			int half = gridResolution / 2;
			// Make sure the new position is still within range
			neighbor.x = neighbor.x < -half ? half : neighbor.x;
			neighbor.y = neighbor.y < -half ? half : neighbor.y;
			neighbor.z = neighbor.z < -half ? half : neighbor.z;

			neighbor.x = neighbor.x > half ? -half : neighbor.x;
			neighbor.y = neighbor.y > half ? -half : neighbor.y;
			neighbor.z = neighbor.z > half ? -half : neighbor.z;

			new_neighbour[i] = gridIndex3Dto1D(neighbor.x, neighbor.y, neighbor.z, gridResolution);
		}
	}

	for (i = 0; i < 8; i++)//In each possible grid, find other bioms that affect this biom
	{
		if ((new_neighbour[i] >= 0) && (new_neighbour[i] != pos_grid))
		{
			start = gridCellStartIndices[new_neighbour[i]];
			end = gridCellEndIndices[new_neighbour[i]];
			/////////////////////////////////////////

			for (j = start; j < end; j++)
			{
				k = particleArrayIndices[j];
				if (k >= 0 && k <= N)
				{
					member = pos[index] - pos[k];
					length = sqrt(member.x*member.x + member.y*member.y + member.z*member.z);

					if (length < rule1Distance)// centralizing effect
					{
						center -= member;
						rule_count[0]++;
					}
					if (length < rule2Distance)// pushing each other if too near
					{
						push += member;
						rule_count[1]++;
					}
					if (length < rule3Distance)//follow neighbour's speed
					{
						speed += vel1[k];
						rule_count[2]++;
					}
				}
			}
			/////////////////////////////////////////
		}
	}

	///////// Compute Bioms in LOCAL Grid ///////////
	start = gridCellStartIndices[pos_grid];
	end = gridCellEndIndices[pos_grid];

	for (j = start; j < end; j++)
	{
		k = particleArrayIndices[j];
		if (k >= 0 && k <= N)
		{
			member = pos[index] - pos[k];
			length = sqrt(member.x*member.x + member.y*member.y + member.z*member.z);

			if (length < rule1Distance)// centralizing effect
			{
				center -= member;
				rule_count[0]++;
			}
			if (length < rule2Distance)// pushing each other if too near
			{
				push += member;
				rule_count[1]++;
			}
			if (length < rule3Distance)//follow neighbour's speed
			{
				speed += vel1[k];
				rule_count[2]++;
			}
		}
	}
	////////////
	//Compute Averange here
	if (rule_count[0] != 0)
	{
		center *= (1.0f / (float)rule_count[0]);
	}
	if (rule_count[1] != 0)
	{
		push *= (1.0f / (float)rule_count[1]);
	}
	if (rule_count[2] != 0)
	{
		speed *= (1.0f / (float)rule_count[2]);
	}
	//local vars
	glm::vec3 out;
	float out_len;
	//Compute new velocity and its speed
	out = vel1[index] + center*rule1Scale + push*rule2Scale + speed*rule3Scale;
	out_len = sqrt(out.x*out.x + out.y*out.y + out.z*out.z);
	
	if (out_len > maxSpeed)//Normalize the speed	
	{
		out = out*(maxSpeed / out_len);
	}
	vel2[index] = out;
}


__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	float range = imax(imax(rule1Distance, rule2Distance), rule3Distance);
	glm::vec3 neighbor;
	int neighbor_num;
	float rate = range / cellWidth;
	glm::vec3 new_pos;
	glm::vec3 a(0.5f);
	a *= gridResolution;
	new_pos = pos[index] * inverseCellWidth + a;
	int pos_grid = gridIndex3Dto1D(new_pos.x, new_pos.y, new_pos.z, gridResolution);
	int i, j;
	float length;
	glm::vec3 len;
	float distance;
	int start, end;
	glm::vec3 member;

	glm::vec3 Vertex[8];//The eight Vertexs of the grid
	Vertex[0] = glm::vec3(0, 0, 0);
	Vertex[1] = glm::vec3(0, 0, 1);
	Vertex[2] = glm::vec3(0, 1, 0);
	Vertex[3] = glm::vec3(0, 1, 1);
	Vertex[4] = glm::vec3(1, 0, 0);
	Vertex[5] = glm::vec3(1, 0, 1);
	Vertex[6] = glm::vec3(1, 1, 0);
	Vertex[7] = glm::vec3(1, 1, 1);

	////////////////////////////
	glm::vec3 center;// center of mass
	glm::vec3 push;// pushing force
	glm::vec3 speed;//velocity of all neighbours
	int rule_count[3];//How many bios are included in each rule
	center = glm::vec3(0);
	push = glm::vec3(0);
	speed = glm::vec3(0);
	rule_count[0] = rule_count[1] = rule_count[2] = 0;
	////////////////////////////////

	int new_neighbour[8];// Possible Grid Index of the neighbouring grids

	for (i = 0; i < 8; i++)// See if this grid is within affection range
	{
		len = new_pos + Vertex[i];//Check each vertex
		distance = sqrt(len.x*len.x + len.y*len.y + len.z*len.z);
		new_neighbour[i] = -1;//Init as -1

		if (distance < range)//if vertex is within range
		{
			neighbor = new_pos + Vertex[i];

			int half = gridResolution / 2;
			// Make sure the new position is still within range
			neighbor.x = neighbor.x < -half ? half : neighbor.x;
			neighbor.y = neighbor.y < -half ? half : neighbor.y;
			neighbor.z = neighbor.z < -half ? half : neighbor.z;

			neighbor.x = neighbor.x > half ? -half : neighbor.x;
			neighbor.y = neighbor.y > half ? -half : neighbor.y;
			neighbor.z = neighbor.z > half ? -half : neighbor.z;

			new_neighbour[i] = gridIndex3Dto1D(neighbor.x, neighbor.y, neighbor.z, gridResolution);
		}
	}

	for (i = 0; i < 8; i++)//In each possible grid, find other bioms that affect this biom
	{
		if ((new_neighbour[i] >= 0) && (new_neighbour[i] != pos_grid))
		{
			start = gridCellStartIndices[new_neighbour[i]];
			end = gridCellEndIndices[new_neighbour[i]];
			/////////////////////////////////////////

			for (j = start; j < end; j++)
			{
				member = pos[index] - pos[j];
				length = sqrt(member.x*member.x + member.y*member.y + member.z*member.z);

				if (length < rule1Distance)// centralizing effect
				{
					center -= member;
					rule_count[0]++;
				}
				if (length < rule2Distance)// pushing each other if too near
				{
					push += member;
					rule_count[1]++;
				}
				if (length < rule3Distance)//follow neighbour's speed
				{
					speed += vel1[j];
					rule_count[2]++;
				}
			}
			/////////////////////////////////////////
		}
	}

	///////// Compute Bioms in LOCAL Grid ///////////
	start = gridCellStartIndices[pos_grid];
	end = gridCellEndIndices[pos_grid];

	for (j = start; j < end; j++)
	{
		member = pos[index] - pos[j];
		length = sqrt(member.x*member.x + member.y*member.y + member.z*member.z);

		if (length < rule1Distance)// centralizing effect
		{
			center -= member;
			rule_count[0]++;
		}
		if (length < rule2Distance)// pushing each other if too near
		{
			push += member;
			rule_count[1]++;
		}
		if (length < rule3Distance)//follow neighbour's speed
		{
			speed += vel1[j];
			rule_count[2]++;
		}

	}
	////////////
	//Compute Averange here
	if (rule_count[0] != 0)
	{
		center *= (1.0f / (float)rule_count[0]);
	}
	if (rule_count[1] != 0)
	{
		push *= (1.0f / (float)rule_count[1]);
	}
	if (rule_count[2] != 0)
	{
		speed *= (1.0f / (float)rule_count[2]);
	}
	//local vars
	glm::vec3 out;
	float out_len;
	//Compute new velocity and its speed
	out = vel1[index] + center*rule1Scale + push*rule2Scale + speed*rule3Scale;
	out_len = sqrt(out.x*out.x + out.y*out.y + out.z*out.z);

	if (out_len > maxSpeed)//Normalize the speed	
	{
		out = out*(maxSpeed / out_len);
	}
	vel2[index] = out;

}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers

	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	//update bioms' velocity,from vel1 to vel2
	kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
	//update bioms' position, from vel2 to pos
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

	//ping-pong vecs by switching pointers rather than values
	glm::vec3* tmp;

	tmp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = tmp;
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	//Reset Grid Buffers
	kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> >(
		numObjects, dev_particleGridIndices, -1);
	kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> >(
		numObjects, dev_particleArrayIndices, -1);
		
	cudaThreadSynchronize();
	//Figure out which grid does this biom occupy, and label	
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(
		numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, 
		dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

	cudaThreadSynchronize();	
	//Thrust and re-arrange the biom index	
	thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);//According to what
	thrust::device_ptr<int> dev_thrust_values(dev_particleArrayIndices);//Arrange who
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, dev_thrust_values);
	
	cudaThreadSynchronize();
	//Identify start and end	
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(
		numObjects, dev_particleGridIndices,dev_particleArrayIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

	cudaThreadSynchronize();
	//Find effective neighbours and Do the rules here	
	kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (
		numObjects,gridSideCount,gridMinimum,
		gridInverseCellWidth,gridCellWidth,dev_gridCellStartIndices,dev_gridCellEndIndices,dev_particleArrayIndices,
		dev_pos,dev_vel1,dev_vel2);

	cudaThreadSynchronize();
	//Update Positions
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel2);	
	//ping-pong vecs by switching pointers rather than values
	glm::vec3* tmp;
	tmp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = tmp;

}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	//Reset Grid Buffers
	kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> >(
		numObjects, dev_particleGridIndices, -1);
	kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> >(
		numObjects, dev_particleArrayIndices, -1);

	cudaThreadSynchronize();
	//Figure out which grid does this biom occupy, and label	
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(
		numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
		dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

	cudaThreadSynchronize();
		
	//Thrust and re-arrange the biom index
	thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);//According to what
	thrust::device_ptr<glm::vec3> dev_thrust_pos(dev_pos);//Arrange pos

	thrust::copy(thrust::device,dev_particleGridIndices, dev_particleGridIndices + numObjects, dev_origin);//Copy grid_index to dev_origin

	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, dev_thrust_pos);//Sort Pos

	thrust::device_ptr<int> dev_thrust_keys2(dev_origin);//According to what
	thrust::device_ptr<glm::vec3> dev_thrust_vel(dev_vel1);//Arrange vel1
	thrust::sort_by_key(dev_thrust_keys2, dev_thrust_keys2 + numObjects, dev_thrust_vel);//Sort Vel1
	
	cudaThreadSynchronize();
	//Identify start and end	
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(
		numObjects, dev_particleGridIndices, dev_particleArrayIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

	cudaThreadSynchronize();
	//Find effective neighbours and Do the rules here	
	
	kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> >(
		numObjects, gridSideCount, gridMinimum,
		gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_pos, dev_vel1, dev_vel2);
		

	cudaThreadSynchronize();
	//Update Positions
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel2);
	//ping-pong vecs by switching pointers rather than values
	glm::vec3* tmp;
	tmp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = tmp;

}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  cudaFree(dev_origin);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  int *intKeys = new int[N];
  int *intValues = new int[N];

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues, sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys, dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues, dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  delete[] intKeys;
  delete[] intValues;
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");

  //system("Pause");

  return;
}
