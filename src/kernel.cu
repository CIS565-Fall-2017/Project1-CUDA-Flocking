#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include <device_launch_parameters.h>
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
#define blockSize 128

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
// These are called ping-pong buffers.
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

glm::vec3 * dev_coherent_pos;
glm::vec3 * dev_coherent_vel;

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
  kernGenerateRandomPosArray<< <fullBlocksPerGrid, blockSize>> >(1, numObjects, dev_pos, scene_scale);
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
  //int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  //int *dev_particleGridIndices; // What grid cell is this particle in?
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  //thrust::device_ptr<int> dev_thrust_particleArrayIndices;
  //thrust::device_ptr<int> dev_thrust_particleGridIndices;
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  //glm::ivec2 * dev_gridCellBounds;
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_coherent_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_coherent_pos failed!");

  cudaMalloc((void**)&dev_coherent_vel, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_coherent_vel failed!");

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

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
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

__forceinline__
__device__ glm::vec3 initializeRule1() 
{
	return glm::vec3(0.f); // center
}

__forceinline__
__device__ glm::vec3 initializeRule2()
{
	return glm::vec3(0.f); // c
}

__forceinline__
__device__ glm::vec3 initializeRule3()
{
	return glm::vec3(0.f); // perceivedVelocity
}

__forceinline__
__device__ glm::vec3 updateRule1(glm::vec3 boidPosition, glm::vec3 neighborPosition, glm::vec3 data) 
{
	return data + neighborPosition;
}

__forceinline__
__device__ glm::vec3 updateRule2(glm::vec3 boidPosition, glm::vec3 neighborPosition, glm::vec3 data)
{
	return data - (neighborPosition - boidPosition);
}

__forceinline__
__device__ glm::vec3 updateRule3(glm::vec3 boidPosition, glm::vec3 neighborVelocity, glm::vec3 data)
{
	return data + neighborVelocity;
}

__forceinline__
__device__ glm::vec3 finishRule3(glm::vec3 boidPosition, int total, glm::vec3 data)
{
	data /= glm::max(1, total);
	return data * rule3Scale;
}

__forceinline__
__device__ glm::vec3 finishRule2(glm::vec3 boidPosition, int total, glm::vec3 data)
{
	return data * rule2Scale;
}

__forceinline__
__device__ glm::vec3 finishRule1(glm::vec3 boidPosition, int total, glm::vec3 data)
{
	data /= glm::max(1, total);
	return (data - boidPosition) * rule1Scale;
}

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int boidIndex, const glm::vec3 *pos, const glm::vec3 *vel) {

	glm::vec3 boidPosition = pos[boidIndex];

	glm::vec3 rule1Data = initializeRule1();
	glm::vec3 rule2Data = initializeRule2();
	glm::vec3 rule3Data = initializeRule3();

	int rule1Total = 0;
	int rule2Total = 0;
	int rule3Total = 0;
	
	for (int b = 0; b < N; ++b)
	{
		glm::vec3 otherBoidPosition = pos[b];

		if (b != boidIndex)
		{
			float d = glm::distance(otherBoidPosition, boidPosition);

			if (d < rule1Distance)
			{
				rule1Data = updateRule1(boidPosition, otherBoidPosition, rule1Data);
				++rule1Total;
			}

			if (d < rule2Distance)
			{
				rule2Data = updateRule2(boidPosition, otherBoidPosition, rule2Data);
				++rule2Total;
			}

			if (d < rule3Distance)
			{
				rule3Data = updateRule3(boidPosition, vel[b], rule3Data);
				++rule3Total;
			}
		}
	}

	glm::vec3 totalVelocity = vel[boidIndex];

	if (rule1Total > 0)
		totalVelocity += finishRule1(boidPosition, rule1Total, rule1Data);

	if (rule2Total > 0)
		totalVelocity += finishRule2(boidPosition, rule2Total, rule2Data);

	if (rule3Total > 0)
		totalVelocity += finishRule3(boidPosition, rule3Total, rule3Data);

	return totalVelocity;
}


__forceinline__
__device__ glm::vec3 clampBoidVelocity(glm::vec3 velocity)
{
	float m = glm::length(velocity);

	if (m > 0.0f)
		return velocity * (glm::clamp(m, 0.f, maxSpeed) / m);

	return velocity;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {

	int boidIndex = threadIdx.x + (blockIdx.x * blockDim.x);

	if (boidIndex >= N)
		return;

	// Compute a new velocity based on pos and vel1
	glm::vec3 resultVelocity = computeVelocityChange(N, boidIndex, pos, vel1);
	resultVelocity = clampBoidVelocity(resultVelocity);

	// Record the new velocity into vel2. Question: why NOT vel1? A: Because we swap velocity buffer indices.
	vel2[boidIndex] = resultVelocity;
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

  //thisPos = modf(thisPos + glm::vec3(scene_scale * .5f), glm::vec3(scene_scale)) - glm::vec3(scene_scale * .5f);
  //thisPos = glm::modf(thisPos + glm::vec3(scene_scale), glm::vec3(scene_scale * 2.f)) - glm::vec3(scene_scale);

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
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__forceinline__
__device__ glm::ivec3 particlePositionToGridIndex(glm::vec3 p, glm::vec3 gridOrigin, float inverseCellWidth)
{
	return glm::floor((p - gridOrigin) * inverseCellWidth);
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) 
{
    // TODO-2.1
	int boidIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	glm::ivec3 cellIndex = particlePositionToGridIndex(pos[boidIndex], gridMin, inverseCellWidth);
	int cellIndex1D = gridIndex3Dto1D(cellIndex.x, cellIndex.y, cellIndex.z, gridResolution);

	if (boidIndex >= N)
		return;

    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
	indices[boidIndex] = boidIndex;
	gridIndices[boidIndex] = cellIndex1D;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) 
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= N)
		return;

	bool start = false;
	bool end = false;
	int gridIndex = particleGridIndices[index];

	// Check start
	if (index == 0)
		start = true;
	else
	{
		int prevGridIndex = particleGridIndices[index - 1];

		if (gridIndex != prevGridIndex)
			start = true;
	}

	if (index == N - 1)
		end = true;
	else
	{
		int nextGridIndex = particleGridIndices[index + 1];

		if (nextGridIndex != gridIndex)
			end = true;
	}

	// TODO-2.1
	// Identify the start point of each cell in the gridIndices array.
	// This is basically a parallel unrolling of a loop that goes
	// "this index doesn't match the one before it, must be a new cell!"
	if (start)
		gridCellStartIndices[gridIndex] = index;

	if (end)
		gridCellEndIndices[gridIndex] = index;
}


__inline__
__device__ bool intersects(glm::vec3 cSphere, float rSphere, glm::vec3 cAABB, float aabbLength)
{
	glm::vec3 minAABB = cAABB;
	glm::vec3 maxAABB = cAABB + glm::vec3(aabbLength);

	glm::vec3 closestPoint = glm::max(minAABB, glm::min(cSphere, maxAABB));
	return glm::length(closestPoint - cSphere) < rSphere;
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) 
{
	// TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
	// the number of boids that need to be checked.
	// - Identify the grid cell that this particle is in
	int boidIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (boidIndex >= N)
		return;

	glm::vec3 boidPosition = pos[boidIndex];
	
	// - Identify which cells may contain neighbors. This isn't always 8.
	// Approach:
	//	- get the AABB of the sphere, iterate only over those cells
	//	- on each cell, check if it actually intersects, to prevent false positives
	float maxRadius = glm::max(rule1Distance, glm::max(rule2Distance, rule3Distance));
	glm::ivec3 fromCell = particlePositionToGridIndex(boidPosition - glm::vec3(maxRadius), gridMin, inverseCellWidth);
	glm::ivec3 toCell = particlePositionToGridIndex(boidPosition + glm::vec3(maxRadius), gridMin, inverseCellWidth);

	fromCell = glm::clamp(fromCell, glm::ivec3(0), glm::ivec3(gridResolution - 1));
	toCell = glm::clamp(toCell, glm::ivec3(0), glm::ivec3(gridResolution - 1));

	glm::vec3 rule1Data = initializeRule1();
	glm::vec3 rule2Data = initializeRule2();
	glm::vec3 rule3Data = initializeRule3();

	int rule1Total = 0;
	int rule2Total = 0;	
	int rule3Total = 0;

	// Loop index order optimized for memory alignment
	for (int z = fromCell.z; z <= toCell.z; ++z)
	{
		for (int y = fromCell.y; y <= toCell.y; ++y)
		{
			for (int x = fromCell.x; x <= toCell.x; ++x)
			{
				glm::vec3 currentCellPosition = glm::vec3(x, y, z) * cellWidth + gridMin;

				// We need to make sure we're actually intersecting with this grid cell
				if (intersects(boidPosition, maxRadius, currentCellPosition, cellWidth))
				{
					int gridIndex = gridIndex3Dto1D(x, y, z, gridResolution);

					int start = gridCellStartIndices[gridIndex];
					int end = gridCellEndIndices[gridIndex];

					// Non empty grid cells
					if (start >= 0 && start < N && end >= 0 && end < N)
					{
						for (int b = start; b <= end; ++b)
						{
							int otherBoidIndex = particleArrayIndices[b];
							glm::vec3 otherBoidPosition = pos[otherBoidIndex];

							if (otherBoidIndex != boidIndex)
							{
								float d = glm::distance(otherBoidPosition, boidPosition);

								if (d < rule1Distance)
								{
									rule1Data = updateRule1(boidPosition, otherBoidPosition, rule1Data);
									++rule1Total;
								}

								if (d < rule2Distance)
								{
									rule2Data = updateRule2(boidPosition, otherBoidPosition, rule2Data);
									++rule2Total;
								}

								if (d < rule3Distance)
								{
									rule3Data = updateRule3(boidPosition, vel1[otherBoidIndex], rule3Data);
									++rule3Total;
								}
							}
						}
					}
				}
			}
		}
	}

	glm::vec3 totalVelocity = vel1[boidIndex];

	if (rule1Total > 0)
		totalVelocity += finishRule1(boidPosition, rule1Total, rule1Data);

	if (rule2Total > 0)
		totalVelocity += finishRule2(boidPosition, rule2Total, rule2Data);

	if (rule3Total > 0)
		totalVelocity += finishRule3(boidPosition, rule3Total, rule3Data);

	// - For each cell, read the start/end indices in the boid pointer array.
	// - Access each boid in the cell and compute velocity change from
	//   the boids rules, if this boid is within the neighborhood distance.
	// - Clamp the speed change before putting the new speed in vel2

	totalVelocity = clampBoidVelocity(totalVelocity);
	vel2[boidIndex] = totalVelocity;
}

__global__ void kernReorderBoidData(int N, int *particleIndices, glm::vec3* pos, 
	glm::vec3 * vel, glm::vec3 * coherentPos, glm::vec3 * coherentVel)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N) 
	{
		int shuffledIndex = particleIndices[index];
		coherentPos[index] = pos[shuffledIndex];
		coherentVel[index] = vel[shuffledIndex];
	}
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 * vel1, glm::vec3 *velCoherent, glm::vec3 * posCoherent) {
	// TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
	// except with one less level of indirection.
	// This should expect gridCellStartIndices and gridCellEndIndices to refer
	// directly to pos and vel1.
	// - Identify the grid cell that this particle is in
	int boidIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (boidIndex >= N)
		return;

	glm::vec3 boidPosition = pos[boidIndex];

	// - Identify which cells may contain neighbors. This isn't always 8.
	// Approach:
	//	- get the AABB of the sphere, iterate only over those cells
	//	- on each cell, check if it actually intersects, to prevent false positives
	float maxRadius = glm::max(rule1Distance, glm::max(rule2Distance, rule3Distance));
	glm::ivec3 fromCell = particlePositionToGridIndex(boidPosition - glm::vec3(maxRadius), gridMin, inverseCellWidth);
	glm::ivec3 toCell = particlePositionToGridIndex(boidPosition + glm::vec3(maxRadius), gridMin, inverseCellWidth);

	fromCell = glm::clamp(fromCell, glm::ivec3(0), glm::ivec3(gridResolution - 1));
	toCell = glm::clamp(toCell, glm::ivec3(0), glm::ivec3(gridResolution - 1));

	glm::vec3 rule1Data = initializeRule1();
	glm::vec3 rule2Data = initializeRule2();
	glm::vec3 rule3Data = initializeRule3();

	int rule1Total = 0;
	int rule2Total = 0;
	int rule3Total = 0;

	// Loop index order optimized for memory alignment
	for (int z = fromCell.z; z <= toCell.z; ++z)
	{
		for (int y = fromCell.y; y <= toCell.y; ++y)
		{
			for (int x = fromCell.x; x <= toCell.x; ++x)
			{
				glm::vec3 currentCellPosition = glm::vec3(x, y, z) * cellWidth + gridMin;

				// We need to make sure we're actually intersecting with this grid cell
				if (intersects(boidPosition, maxRadius, currentCellPosition, cellWidth))
				{
					// - For each cell, read the start/end indices in the boid pointer array.
					int gridIndex = gridIndex3Dto1D(x, y, z, gridResolution);

					int start = gridCellStartIndices[gridIndex];
					int end = gridCellEndIndices[gridIndex];

					// Non empty grid cells
					if (start >= 0 && start < N && end >= 0 && end < N)
					{
						for (int b = start; b <= end; ++b)
						{
							//   DIFFERENCE: For best results, consider what order the cells should be
							//   checked in to maximize the memory benefits of reordering the boids data.
							// - Access each boid in the cell and compute velocity change from
							//   the boids rules, if this boid is within the neighborhood distance.
							glm::vec3 otherBoidPosition = posCoherent[b];
							glm::vec3 otherBoidVelocity = velCoherent[b];

							if (b != boidIndex)
							{
								float d = glm::distance(otherBoidPosition, boidPosition);

								if (d < rule1Distance)
								{
									rule1Data = updateRule1(boidPosition, otherBoidPosition, rule1Data);
									++rule1Total;
								}

								if (d < rule2Distance)
								{
									rule2Data = updateRule2(boidPosition, otherBoidPosition, rule2Data);
									++rule2Total;
								}

								if (d < rule3Distance)
								{
									rule3Data = updateRule3(boidPosition, otherBoidVelocity, rule3Data);
									++rule3Total;
								}
							}
						}
					}
				}
			}
		}
	}

	glm::vec3 totalVelocity = vel1[boidIndex];

	if (rule1Total > 0)
		totalVelocity += finishRule1(boidPosition, rule1Total, rule1Data);

	if (rule2Total > 0)
		totalVelocity += finishRule2(boidPosition, rule2Total, rule2Data);

	if (rule3Total > 0)
		totalVelocity += finishRule3(boidPosition, rule3Total, rule3Data);
	
	// - Clamp the speed change before putting the new speed in vel2
	totalVelocity = clampBoidVelocity(totalVelocity);
	vel1[boidIndex] = totalVelocity;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	// TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel1);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

	// TODO-1.2 ping-pong the velocity buffers
	std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt) {
	
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	// TODO-2.1
	// Uniform Grid Neighbor search using Thrust sort.
	// In Parallel:
	// - label each particle with its array index as well as its grid index.
	//   Use 2x width grids.
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("kernComputeIndices failed!");
	
	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer failed!");
	kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer failed!");

	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	// - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

	// - Update positions
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel1);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	// - Ping-pong buffers as needed
	std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationCoherentGrid(float dt) 
{
	// TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
	// Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	// In Parallel:
	// - Label each particle with its array index as well as its grid index.
	//   Use 2x width grids
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("kernComputeIndices failed!");

	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer failed!");
	kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer failed!");

	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	// - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
	//   the particle data in the simulation array.
	//   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
	kernReorderBoidData << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_coherent_pos, dev_coherent_vel);
	checkCUDAErrorWithLine("kernReorderBoidData failed!");

	// - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos, dev_vel1, dev_coherent_vel, dev_coherent_pos);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

	// - Update positions
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel1);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	// - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
	//std::swap(dev_vel1, dev_vel2); // No need to ping pong! We always copy vel1 into velCoherent
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

  cudaFree(dev_coherent_pos);
  cudaFree(dev_coherent_vel);
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
  return;
}
