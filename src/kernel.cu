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

//#define print 0

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
#define rule1Distance 1.0f
#define rule2Distance 0.6f
#define rule3Distance 1.0f

#define rule1Scale 0.1f
#define rule2Scale 1.0f
#define rule3Scale 1.0f

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
glm::vec3 *dev_pingPongPos;

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

  cudaMalloc((void**)&dev_pingPongPos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale);
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

  // DONE-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("failed to allocate indicies for uniform grid");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("failed to allocate grid index buffers");

  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
 
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

  kernCopyPositionsToVBO <<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO <<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

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
	glm::vec3 perceived_center = glm::vec3(0.0f,0.0f,0.0f);
	glm::vec3 flee_direction = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 perceived_vel = glm::vec3(0.0f, 0.0f, 0.0f);

	glm::vec3 output;

	float distance;
	int rule1neighbors = 0;
	int rule3neighbors = 0;

	for (int i = 0; i < N; i++) {
		if (i != iSelf) {
			distance = glm::distance(pos[i], pos[iSelf]);

			// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
			if (distance < rule1Distance) {
				perceived_center += pos[i];
				rule1neighbors++;
			}
			// Rule 2: boids try to stay a distance d away from each other
			if (distance < rule2Distance) flee_direction -= (pos[i] - pos[iSelf]);
			// Rule 3: boids try to match the speed of surrounding boids
			if (distance < rule3Distance) {
				perceived_vel += (vel[i]);
				rule3neighbors++;
			}
		}
	}

	if (rule1neighbors > 0) {
		perceived_center /= (rule1neighbors);
		output += (perceived_center - pos[iSelf]) * rule1Scale;
	}

	if (rule3neighbors > 0) {
		perceived_vel /= (rule3neighbors);
		output += perceived_vel * rule3Scale;
	}

	return output + flee_direction * rule2Scale;
}

/**
* DONE-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) return;
  // Compute a new velocity based on pos and vel1
	glm::vec3 thisVel = vel1[index] + computeVelocityChange(N, index, pos, vel1);
  // Clamp the speed
	if (glm::length(thisVel) <= maxSpeed) vel2[index] = thisVel;
	else vel2[index] = glm::normalize(thisVel) * maxSpeed;
  // Record the new velocity into vel2. Question: why NOT vel1?
	//ANSWER FOR README we reference vel1 for a snapshot of that step; we don't want
	//to consider velocities for step n+1 during step n
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
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // DONE-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) return;

	//index represents this boid's current memory address in PAI, vel, and pos
	indices[index] = index;

	//grid index is the floored result of pos/gridWidth, will need to be divided again for remainder (octant check)
	glm::ivec3 relativePos = (pos[index] - gridMin) * inverseCellWidth;
	gridIndices[index] = gridIndex3Dto1D(relativePos[0], relativePos[1], relativePos[2], gridResolution);
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
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // DONE-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) return;

	int thisPGI = particleGridIndices[index];

	if (index == 0) {
		gridCellStartIndices[thisPGI] = index;
		return;
	}

	if (index == N - 1) {
		gridCellEndIndices[thisPGI] = index;
	}

	//if bird[index] != bird[index-1], then bird[index-1] is the last of its cell and bird[index] is the first of its cell
	if (particleGridIndices[index - 1] != thisPGI) {
		gridCellStartIndices[thisPGI] = index;
		gridCellEndIndices[particleGridIndices[index - 1]] = index - 1;
	}
	
	//if there are no boids in a cell, it should be untouched and remain -1
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // DONE-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) return;

  // - Identify the grid cell that this particle is in
	//If the quotient of position/gridwidth for any dimension is more than halfway
	//to the next cell, we know it's in the far quadrant
	glm::vec3 gridCellIndicesRaw = (pos[index] - gridMin) * inverseCellWidth;
	glm::ivec3 gridCellIndices = gridCellIndicesRaw;
	glm::vec3 gridCellOverflow = gridCellIndicesRaw - glm::vec3(gridCellIndices);

	//determine if the overflow is > 0.5, but then make sure that the desired neighbor is legal
	int xNeighbor = gridCellIndices.x + (gridCellOverflow.x > 0.5f ? 1 : -1);
	if (xNeighbor < 0 || xNeighbor > gridResolution) xNeighbor = 0;
	int yNeighbor = gridCellIndices.y + gridCellOverflow.y > 0.5f ? 1 : -1;
	if (yNeighbor < 0 || yNeighbor > gridResolution) yNeighbor = 0;
	int zNeighbor = gridCellIndices.z + gridCellOverflow.z > 0.5f ? 1 : -1;
	if (zNeighbor < 0 || zNeighbor > gridResolution) zNeighbor = 0;

  // - Identify which cells may contain neighbors. This isn't always 8.
	int neighbors = 0;
	int neighborIndices[8];

	//self is always legal
	neighborIndices[neighbors++] = gridIndex3Dto1D(gridCellIndices.x, gridCellIndices.y, gridCellIndices.z, gridResolution);

	if (xNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(xNeighbor, gridCellIndices.y, gridCellIndices.z, gridResolution);
	if (yNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(gridCellIndices.x, yNeighbor, gridCellIndices.z, gridResolution);
	if (zNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(gridCellIndices.x, gridCellIndices.y, zNeighbor, gridResolution);;
	if (xNeighbor && yNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(xNeighbor, yNeighbor, gridCellIndices.z, gridResolution);
	if (xNeighbor && zNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(xNeighbor, gridCellIndices.y, zNeighbor, gridResolution);
	if (yNeighbor && zNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(gridCellIndices.x, yNeighbor, zNeighbor, gridResolution);
	if (xNeighbor && yNeighbor && zNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(xNeighbor, yNeighbor, zNeighbor, gridResolution);

	glm::vec3 perceived_center = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 flee_direction = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 perceived_vel = glm::vec3(0.0f, 0.0f, 0.0f);

	glm::vec3 output;

	float distance;
	int rule1neighbors = 0;
	int rule3neighbors = 0;

  // - For each cell, read the start/end indices in the boid pointer array.
	for (int i = 0; i < neighbors; i++) {
		int gridCellIndex = neighborIndices[i];
		for (int j = gridCellStartIndices[gridCellIndex]; j <= gridCellEndIndices[gridCellIndex] && j > -1; j++) {
			// - Access each boid in the cell and compute velocity change from
			//   the boids rules, if this boid is within the neighborhood distance.
			int thisIndex = particleArrayIndices[j];
			if (thisIndex != index) {
				distance = glm::distance(pos[thisIndex], pos[index]);

				// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
				if (distance < rule1Distance) {
					perceived_center += pos[thisIndex];
					rule1neighbors++;
				}
				// Rule 2: boids try to stay a distance d away from each other
				if (distance < rule2Distance) flee_direction -= (pos[thisIndex] - pos[index]);
				// Rule 3: boids try to match the speed of surrounding boids
				if (distance < rule3Distance) {
					perceived_vel += (vel1[thisIndex]);
					rule3neighbors++;
				}
			}
		}
	}

	if (rule1neighbors > 0) {
		perceived_center /= (rule1neighbors);
		output += (perceived_center - pos[index]) * rule1Scale;
	}

	if (rule3neighbors > 0) {
		perceived_vel /= (rule3neighbors);
		output += perceived_vel * rule3Scale;
	}

	output = vel1[index] + output + flee_direction * rule2Scale;
	// Clamp the speed and put into vel2
	if (glm::length(output) <= maxSpeed) vel2[index] = output;
	else vel2[index] = glm::normalize(output) * maxSpeed;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) return;

	// - Identify the grid cell that this particle is in
	//If the quotient of position/gridwidth for any dimension is more than halfway
	//to the next cell, we know it's in the far quadrant
	glm::vec3 gridCellIndicesRaw = (pos[index] - gridMin) * inverseCellWidth;
	glm::ivec3 gridCellIndices = gridCellIndicesRaw;
	glm::vec3 gridCellOverflow = gridCellIndicesRaw - glm::vec3(gridCellIndices);

	//determine if the overflow is > 0.5, but then make sure that the desired neighbor is legal
	int xNeighbor = gridCellIndices.x + (gridCellOverflow.x > 0.5f ? 1 : -1);
	if (xNeighbor < 0 || xNeighbor > gridResolution) xNeighbor = 0;
	int yNeighbor = gridCellIndices.y + gridCellOverflow.y > 0.5f ? 1 : -1;
	if (yNeighbor < 0 || yNeighbor > gridResolution) yNeighbor = 0;
	int zNeighbor = gridCellIndices.z + gridCellOverflow.z > 0.5f ? 1 : -1;
	if (zNeighbor < 0 || zNeighbor > gridResolution) zNeighbor = 0;

	// - Identify which cells may contain neighbors. This isn't always 8.
	int neighbors = 0;
	int neighborIndices[8];

	//self is always legal
	neighborIndices[neighbors++] = gridIndex3Dto1D(gridCellIndices.x, gridCellIndices.y, gridCellIndices.z, gridResolution);

	if (xNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(xNeighbor, gridCellIndices.y, gridCellIndices.z, gridResolution);
	if (yNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(gridCellIndices.x, yNeighbor, gridCellIndices.z, gridResolution);
	if (zNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(gridCellIndices.x, gridCellIndices.y, zNeighbor, gridResolution);;
	if (xNeighbor && yNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(xNeighbor, yNeighbor, gridCellIndices.z, gridResolution);
	if (xNeighbor && zNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(xNeighbor, gridCellIndices.y, zNeighbor, gridResolution);
	if (yNeighbor && zNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(gridCellIndices.x, yNeighbor, zNeighbor, gridResolution);
	if (xNeighbor && yNeighbor && zNeighbor) neighborIndices[neighbors++] = gridIndex3Dto1D(xNeighbor, yNeighbor, zNeighbor, gridResolution);

	glm::vec3 perceived_center = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 flee_direction = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 perceived_vel = glm::vec3(0.0f, 0.0f, 0.0f);

	glm::vec3 output;

	float distance;
	int rule1neighbors = 0;
	int rule3neighbors = 0;

	// - For each cell, read the start/end indices in the boid pointer array.
	for (int i = 0; i < neighbors; i++) {
		int gridCellIndex = neighborIndices[i];
		for (int j = gridCellStartIndices[gridCellIndex]; j <= gridCellEndIndices[gridCellIndex] && j > -1; j++) {
			// - Access each boid in the cell and compute velocity change from
			//   the boids rules, if this boid is within the neighborhood distance.
			if (j != index) {
				distance = glm::distance(pos[j], pos[index]);

				// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
				if (distance < rule1Distance) {
					perceived_center += pos[j];
					rule1neighbors++;
				}
				// Rule 2: boids try to stay a distance d away from each other
				if (distance < rule2Distance) flee_direction -= (pos[j] - pos[index]);
				// Rule 3: boids try to match the speed of surrounding boids
				if (distance < rule3Distance) {
					perceived_vel += (vel1[j]);
					rule3neighbors++;
				}
			}
		}
	}

	if (rule1neighbors > 0) {
		perceived_center /= (rule1neighbors);
		output += (perceived_center - pos[index]) * rule1Scale;
	}

	if (rule3neighbors > 0) {
		perceived_vel /= (rule3neighbors);
		output += perceived_vel * rule3Scale;
	}

	output = vel1[index] + output + flee_direction * rule2Scale;
	// Clamp the speed and put into vel2
	if (glm::length(output) <= maxSpeed) vel2[index] = output;
	else vel2[index] = glm::normalize(output) * maxSpeed;

}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernUpdateVelocityBruteForce <<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("Brute force calls failed!");

	kernUpdatePos <<<fullBlocksPerGrid, blockSize>>> (numObjects, dt, dev_pos, dev_vel1);
	checkCUDAErrorWithLine("Position Update calls failed!");

  // TODO-1.2 ping-pong the velocity buffers
	glm::vec3* temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = temp;

}

void Boids::stepSimulationScatteredGrid(float dt) {

  // DONE-2.1
  // Uniform Grid Neighbor search using Thrust sort.

	dim3 blocksToMakeN((numObjects + blockSize - 1) / blockSize); //number of blocks needed to iterate over numObjects
	dim3 blocksToMakeCells((gridCellCount + blockSize - 1) / blockSize); //number of blocks to iterate over gridCellCount

	//reset startend arrays, where -1 means that no boids are in that cell
	kernResetIntBuffer <<<blocksToMakeCells, blockSize>>> (gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer <<<blocksToMakeCells, blockSize>>> (gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("Failed to Reset Arrays to -1");

  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
	kernComputeIndices <<<blocksToMakeN, blockSize>>> (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("Failed to Compute Grid Indicies");

  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
	checkCUDAErrorWithLine("Failed to Sort Grid Indicies");

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd<<<blocksToMakeN, blockSize>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("Failed to Determine Start/End points");

  // - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchScattered <<<blocksToMakeN, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("Failed to comput velocity (scattered)");

  // - Update positions
	kernUpdatePos<<<blocksToMakeN,blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("Failed to Update Positions (scattered)");

  // - Ping-pong buffers as needed
	glm::vec3* temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = temp;
}

//helper for coherent step
__global__ void kernReorderForCoherent(int N, glm::vec3* pos, glm::vec3* pingPongPos, glm::vec3* vel1, glm::vec3* vel2, int* arrayIndices) {
	//we'll ping pong both our vel and pos to newly ordered ones. We'll switch them to the expected addresses after this in the Step
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) return;

	pingPongPos[index] = pos[arrayIndices[index]];
	vel2[index] = vel1[arrayIndices[index]];
}

void Boids::stepSimulationCoherentGrid(float dt) {
	dim3 blocksToMakeN((numObjects + blockSize - 1) / blockSize); //number of blocks needed to iterate over numObjects
	dim3 blocksToMakeCells((gridCellCount + blockSize - 1) / blockSize); //number of blocks to iterate over gridCellCount

																		 //reset startend arrays, where -1 means that no boids are in that cell
	kernResetIntBuffer << <blocksToMakeCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <blocksToMakeCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("Failed to Reset Arrays to -1");

	// In Parallel:
	// - label each particle with its array index as well as its grid index.
	//   Use 2x width grids.
	kernComputeIndices << <blocksToMakeN, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("Failed to Compute Grid Indicies");

	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
	checkCUDAErrorWithLine("Failed to Sort Grid Indicies");

	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd<<<blocksToMakeN, blockSize>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("Failed to Determine Start/End points");

	kernReorderForCoherent<<<blocksToMakeN,blockSize>>>(numObjects, dev_pos, dev_pingPongPos, dev_vel1, dev_vel2, dev_particleArrayIndices);
	checkCUDAErrorWithLine("Failed to reorder boid data");

	//at this point, pos contains incorrectly ordered data. We'll give that to pos
	glm::vec3* temp = dev_pos;
	dev_pos = dev_pingPongPos;
	dev_pingPongPos = temp;

	//likwise, vel2 has the good stuff, but we're used to vel1 having it.
	temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = temp;

	//this is redundant, we could accommodate these new locations... But for intelligibility, we'll exchange these extra 4 pointers every frame.
	
	// - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchScattered << <blocksToMakeN, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("Failed to comput velocity (scattered)");

	// - Update positions
	kernUpdatePos << <blocksToMakeN, blockSize >> >(numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("Failed to Update Positions (scattered)");

	// - Ping-pong buffers as needed
	temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = temp;
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  cudaFree(dev_pingPongPos);
  
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
}
