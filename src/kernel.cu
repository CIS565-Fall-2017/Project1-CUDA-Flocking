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
glm::vec3 *dev_pos_sorted;
glm::vec3 *dev_vel_sorted;

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
	a = (a ^ 0xc761c23c) ^ (a >> 20);
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
	kernGenerateRandomPosArray << <fullBlocksPerGrid, blockSize >> >(1, numObjects,
		dev_pos, scene_scale);
	checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

	// LOOK-2.1 computing grid params
	gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
	//Uncomment following line and comment the above line if you want to test the efficiency of searching neighbours in 27 cells
	//gridCellWidth = std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
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

	cudaMalloc((void**)&dev_pos_sorted, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_vel_sorted, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

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
	glm::vec3 rule1V, rule2V, rule3V;
	glm::vec3 perceived_center;
	int count1 = 0, count3 = 0;
	for (int i = 0; i < N; i++) {
		if (i == iSelf) continue;
		float dis = glm::length(pos[i] - pos[iSelf]);
		if (dis < rule1Distance) {
			perceived_center += pos[i];
			count1++;
		}
		if (dis < rule2Distance) 
			rule2V -= (pos[i] - pos[iSelf]);
		if (dis < rule3Distance) {
			rule3V += vel[i];
			count3++;
		}
	}
	if (count1) {
		perceived_center /= count1;
		rule1V= (perceived_center-pos[iSelf]) * rule1Scale;
	}
	rule2V *= rule2Scale;
	if (count3) {
		rule3V /= count3;
		rule3V = (rule3V - vel[iSelf]) * rule3Scale;
	}
	return rule1V + rule2V + rule3V;

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
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
		return;
	vel2[index] = vel1[index] + computeVelocityChange(N, index, pos, vel1);
	vel2[index] = glm::length(vel2[index]) <= maxSpeed ? vel2[index]:glm::normalize(vel2[index]) * maxSpeed;
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
	//printf("Pos index:pos[%d]:%f %f %f\n",index, pos[index].x, pos[index].y, pos[index].z);
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
	// TODO-2.1
	// - Label each boid with the index of its grid cell.
	// - Set up a parallel array of integer indices as pointers to the actual
	//   boid data in pos and vel1/vel2
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)	return;
	int x = (int)((pos[index].x - gridMin.x) * inverseCellWidth);
	int y = (int)((pos[index].y - gridMin.y) * inverseCellWidth);
	int z = (int)((pos[index].z - gridMin.z) * inverseCellWidth);
	gridIndices[index] = gridIndex3Dto1D(x, y, z, gridResolution);
	indices[index] = index;
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
	// TODO-2.1
	// Identify the start point of each cell in the gridIndices array.
	// This is basically a parallel unrolling of a loop that goes
	// "this index doesn't match the one before it, must be a new cell!"
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) return;
	if (index == 0) gridCellStartIndices[particleGridIndices[0]] = 0;
	else if (index == N - 1) {
		gridCellEndIndices[particleGridIndices[N - 1]] = N - 1;
		return;
	}
	if (particleGridIndices[index] != particleGridIndices[index + 1]) {
		gridCellEndIndices[particleGridIndices[index]] = index;
		gridCellStartIndices[particleGridIndices[index + 1]] = index + 1;
	}

}

__global__ void kernUpdateVelNeighborSearchScattered(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int *gridCellStartIndices, int *gridCellEndIndices,
	int *particleArrayIndices, int *particleGridIndices,
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
	if (index >=N)
		return;
	int particleIdx = particleArrayIndices[index];
	int gridIndex = particleGridIndices[index];
	int GridCellCount= gridResolution * gridResolution * gridResolution;
	float x = (pos[particleIdx].x - gridMin.x)*inverseCellWidth;
	float y= (pos[particleIdx].y - gridMin.y)*inverseCellWidth;
	float z = (pos[particleIdx].z - gridMin.z)*inverseCellWidth;
	int quadrantX = int(x + 0.5) - int(x) ? 1 : 0;
	int quadrantY = int(y + 0.5) - int(y) ? 1 : 0;
	int quadrantZ = int(z + 0.5) - int(z) ? 1 : 0;
	glm::vec3 rule1V, rule2V, rule3V;
	glm::vec3 perceived_center;
	int count1 = 0, count3 = 0;
	//For 8 cell neighbour search, we need to find out which quadrant the current particle is in inside the cell
	for (int i = quadrantX - 1; i <= quadrantX; i++) {
		for (int j = quadrantY - 1; j <= quadrantY; j++) {
			for (int k = quadrantZ - 1; k <= quadrantZ; k++) {
				int GID = gridIndex3Dto1D((int)x + i, (int)y + j, (int)z + k, gridResolution);
				if (GID < 0 || GID >= GridCellCount)
					continue;
				int startIndex = gridCellStartIndices[GID];
				int endIndex = gridCellEndIndices[GID];
				if (startIndex == -1 || endIndex == -1)
					continue;
				for (int n = startIndex; n <= endIndex; n++) {
					int idx = particleArrayIndices[n];
					if (idx == particleIdx) continue;
					float dis = glm::length(pos[idx] - pos[particleIdx]);
					if (dis < rule1Distance) {
						perceived_center += pos[idx];
						count1++;
					}
					if (dis < rule2Distance)
						rule2V -= (pos[idx] - pos[particleIdx]);
					if (dis < rule3Distance) {
						rule3V += vel1[idx];
						count3++;
					}
				}
			}
		}
	}

	//Uncomment the following loop and comment the above loop if you want to test the efficiency of searching neighbours in 27 cells
	//Also, change the gridCellWidth to be the length of the searching distance instead of double of it. 
	/*for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				int GID = gridIndex3Dto1D((int)x + i, (int)y + j, (int)z + k, gridResolution);
				if (GID < 0 || GID >= GridCellCount)
					continue;
				int startIndex = gridCellStartIndices[GID];
				int endIndex = gridCellEndIndices[GID];
				if (startIndex == -1 || endIndex == -1)
					continue;
				for (int n = startIndex; n <= endIndex; n++) {
					int idx = particleArrayIndices[n];
					if (idx == particleIdx) continue;
					float dis = glm::length(pos[idx] - pos[particleIdx]);
					if (dis < rule1Distance) {
						perceived_center += pos[idx];
						count1++;
					}
					if (dis < rule2Distance)
						rule2V -= (pos[idx] - pos[particleIdx]);
					if (dis < rule3Distance) {
						rule3V += vel1[idx];
						count3++;
					}
				}
			}
		}
	}*/

	if (count1) {
		perceived_center /= count1;
		rule1V = (perceived_center - pos[particleIdx]) * rule1Scale;
	}
	rule2V *= rule2Scale;
	if (count3) {
		rule3V /= count3;
		rule3V = (rule3V - vel1[particleIdx]) * rule3Scale;
	}
	vel2[particleIdx] = vel1[particleIdx] + rule1V + rule2V + rule3V;
	vel2[particleIdx] = glm::length(vel2[particleIdx]) <= maxSpeed ? vel2[particleIdx] : glm::normalize(vel2[particleIdx]) * maxSpeed;

}

__global__ void kernUpdateVelNeighborSearchCoherent(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int *gridCellStartIndices, int *gridCellEndIndices, int *particleGridIndices,
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
	if (index >= N)
		return;
	int gridIndex = particleGridIndices[index];
	int GridCellCount = gridResolution * gridResolution * gridResolution;
	float x = (pos[index].x - gridMin.x)*inverseCellWidth;
	float y = (pos[index].y - gridMin.y)*inverseCellWidth;
	float z = (pos[index].z - gridMin.z)*inverseCellWidth;
	int quadrantX = int(x + 0.5) - int(x) ? 1 : 0;
	int quadrantY = int(y + 0.5) - int(y) ? 1 : 0;
	int quadrantZ = int(z + 0.5) - int(z) ? 1 : 0;
	glm::vec3 rule1V, rule2V, rule3V;
	glm::vec3 perceived_center;
	int count1 = 0, count3 = 0;
	for (int i = quadrantX - 1; i <= quadrantX; i++) {
		for (int j = quadrantY - 1; j <= quadrantY; j++) {
			for (int k = quadrantZ - 1; k <= quadrantZ; k++) {
				int GID = gridIndex3Dto1D((int)x + i, (int)y + j, (int)z + k, gridResolution);
				if (GID < 0 || GID >= GridCellCount)
					continue;
				int startIndex = gridCellStartIndices[GID];
				int endIndex = gridCellEndIndices[GID];
				if (startIndex == -1 || endIndex == -1)
					continue;
				for (int n = startIndex; n <= endIndex; n++) {
					if (n== index) continue;
					float dis = glm::length(pos[n] - pos[index]);
					if (dis < rule1Distance) {
						perceived_center += pos[n];
						count1++;
					}
					if (dis < rule2Distance)
						rule2V -= (pos[n] - pos[index]);
					if (dis < rule3Distance) {
						rule3V += vel1[n];
						count3++;
					}
				}
			}
		}
	}
	//Uncomment the following loop and comment the above loop if you want to test the efficiency of searching neighbours in 27 cells
	//Also, change the gridCellWidth to be the length of the searching distance instead of double of it. 
	/*for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				int GID = gridIndex3Dto1D((int)x + i, (int)y + j, (int)z + k, gridResolution);
				if (GID < 0 || GID >= GridCellCount)
					continue;
				int startIndex = gridCellStartIndices[GID];
				int endIndex = gridCellEndIndices[GID];
				if (startIndex == -1 || endIndex == -1)
					continue;
				for (int n = startIndex; n <= endIndex; n++) {
					if (n== index) continue;
					float dis = glm::length(pos[n] - pos[index]);
					if (dis < rule1Distance) {
						perceived_center += pos[n];
						count1++;
					}
					if (dis < rule2Distance)
						rule2V -= (pos[n] - pos[index]);
					if (dis < rule3Distance) {
						rule3V += vel1[n];
						count3++;
					}
				}
			}
		}
	}*/
	if (count1) {
		perceived_center /= count1;
		rule1V = (perceived_center - pos[index]) * rule1Scale;
	}
	rule2V *= rule2Scale;
	if (count3) {
		rule3V /= count3;
		rule3V = (rule3V - vel1[index]) * rule3Scale;
	}
	vel2[index] = vel1[index] + rule1V + rule2V + rule3V;
	vel2[index] = glm::length(vel2[index]) <= maxSpeed ? vel2[index] : glm::normalize(vel2[index]) * maxSpeed;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	// TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	// TODO-1.2 ping-pong the velocity buffers
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, dev_vel1, dev_vel2);
	//cudaThreadSynchronize();
	std::swap(dev_vel1, dev_vel2);
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel1);
	//cudaThreadSynchronize();
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
	dim3 fullBlocksPerGridForCells((gridCellCount + blockSize - 1) / blockSize);
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos,
		dev_particleArrayIndices, dev_particleGridIndices);

	dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
	dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
	kernResetIntBuffer << <fullBlocksPerGridForCells, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <fullBlocksPerGridForCells, blockSize >> >(gridCellCount, dev_gridCellEndIndices, -1);
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
		gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_particleGridIndices, dev_pos, dev_vel1, dev_vel2);
	std::swap(dev_vel1, dev_vel2);
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel1);
}

__global__ void kernResortParticleData(int N, int* particleArrayIndices, glm::vec3* pos_sorted, glm::vec3* pos, glm::vec3* vel_sorted, glm::vec3* vel)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index >= N) return;
	int particleIdx = particleArrayIndices[index];
	pos_sorted[index] = pos[particleIdx];
	vel_sorted[index] = vel[particleIdx];

}

void Boids::stepSimulationCoherentGrid(float dt,float* Neighbour_Search_T) {
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
	dim3 fullBlocksPerGridForCells((gridCellCount + blockSize - 1) / blockSize);
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos,
		dev_particleArrayIndices, dev_particleGridIndices);

	dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleGridIndices);
	dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
	thrust::sort_by_key(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects, dev_thrust_particleGridIndices);
	kernResetIntBuffer << <fullBlocksPerGridForCells, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <fullBlocksPerGridForCells, blockSize >> >(gridCellCount, dev_gridCellEndIndices, -1);
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	
	//Resort the boids data to make they are contiguous if they are belong to the same cell
	kernResortParticleData << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleArrayIndices, dev_pos_sorted, dev_pos, dev_vel_sorted, dev_vel1);
	std::swap(dev_pos, dev_pos_sorted);
	
	//Coherent Neighbour searching
	//Use these to test the efficiency of the NeighborSearch
	/*cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);*/

	kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
		gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleGridIndices, dev_pos, dev_vel_sorted, dev_vel1);
	
	//cudaEventRecord(stop1, NULL);
	//cudaEventSynchronize(stop1);
	////Here we just record the searching time for the initial boids distribution.
	//if ((*Neighbour_Search_T) == 0)
	//	cudaEventElapsedTime(Neighbour_Search_T, start1, stop1);

	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel1);
}

void Boids::endSimulation() {
	cudaFree(dev_vel1);
	cudaFree(dev_vel2);
	cudaFree(dev_pos);
	// TODO-2.1 TODO-2.3 - Free any additional buffers here.
	cudaFree(dev_gridCellEndIndices);
	cudaFree(dev_gridCellStartIndices);
	cudaFree(dev_particleArrayIndices);
	cudaFree(dev_particleGridIndices);
	cudaFree(dev_pos_sorted);
	cudaFree(dev_vel_sorted);
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
