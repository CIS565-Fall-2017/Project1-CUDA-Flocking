#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// Useful for doing grid-based neighbor search
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

// Parameters for the boids algorithm.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

//Size of the starting area in simulation space. --> -scene_scale to scene_scale in every dimension
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// These buffers are here to hold all your boid information.
// These are allocated in Boids::initSimulation.
// We need two velocity buffers in a simulation so we can ping-pong the buffers.
// This way we can modify the data of one of the velocity buffers while reading from the other
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // Stores which index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // Stores which grid cell is this particle in?

							  // Needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // Stores Which part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this grid cell?

							   // Additional buffers needed to reshuffle
							   // the position and velocity data to be coherent within cells.
glm::vec3 *dev_coherentVel; //rearranged form of dev_vel2 so that it is more memory coherent
glm::vec3 *dev_coherentPos; //rearranged form of dev_pos so that it is more memory coherent

							// Grid parameters based on simulation parameters.
							// These are computed in Boids::initSimulation
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

/*
* Helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index)
{
	thrust::default_random_engine rng(hash((int)(index * time)));
	thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

	return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* This is a basic CUDA kernel.
* CUDA kernel for generating boids with a random position somewhere inside the simulation space
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale)
{
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
void Boids::initSimulation(int N)
{
	numObjects = N;
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize); //To ensure if N is not an exact multiple of blocksize, 
															 //the remainder of N/blocksize is still a portion of N which 
															 //would be ignored if we dont have an extra block to 
															 //accommodate the remainder of the N objects

															 // Basic CUDA memory management and error checking.
															 // Don't forget to cudaFree in  Boids::endSimulation.
															 // Allocating all buffers at once is more efficient.
	cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

	cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

	//generate random initial positions for boids
	kernGenerateRandomPosArray << <fullBlocksPerGrid, blockSize >> >(1, numObjects, dev_pos, scene_scale);
	checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

	//Computing grid parameters
	gridCellWidth = 1.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
	int halfSideCount = (int)(scene_scale / gridCellWidth) + 1; //not sure why + 1
	gridSideCount = 2 * halfSideCount;

	gridCellCount = gridSideCount * gridSideCount * gridSideCount;
	gridInverseCellWidth = 1.0f / gridCellWidth;
	float halfGridWidth = gridCellWidth * halfSideCount;
	gridMinimum.x -= halfGridWidth;
	gridMinimum.y -= halfGridWidth;
	gridMinimum.z -= halfGridWidth;

	cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

	cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

	cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

	cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

	cudaMalloc((void**)&dev_coherentPos, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_coherentPos failed!");

	cudaMalloc((void**)&dev_coherentVel, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_coherentVel failed!");

	cudaThreadSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/*
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * index + 0] = vel[index].x + 0.3f;
		vbo[4 * index + 1] = vel[index].y + 0.3f;
		vbo[4 * index + 2] = vel[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}

/*
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities)
{
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
* Helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity of the boid with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel)
{
	glm::vec3 v1 = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 v2 = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 v3 = glm::vec3(0.0f, 0.0f, 0.0f);

	glm::vec3 percieved_center_of_mass = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 perceived_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 separate_vector = glm::vec3(0.0f, 0.0f, 0.0f);

	int neighborCount1 = 0;
	int neighborCount3 = 0;

	float distance = 0.0f;

	for (int i = 0; i < N; i++)
	{
		if (i != iSelf)
		{
			// 3 rules for the basic boids algorithm
			distance = glm::distance(pos[i], pos[iSelf]);
			if (distance < rule1Distance)
			{
				percieved_center_of_mass += pos[i];
				neighborCount1++;
			}

			if (distance < rule2Distance)
			{
				separate_vector -= (pos[i] - pos[iSelf]);
			}

			if (distance < rule3Distance)
			{
				perceived_velocity += vel[i];
				neighborCount3++;
			}
		}
	}

	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	if (neighborCount1 != 0)
	{
		percieved_center_of_mass /= neighborCount1;
		v1 = (percieved_center_of_mass - pos[iSelf])*rule1Scale;
	}

	// Rule 2: boids try to stay a distance d away from each other
	v2 = separate_vector*rule2Scale;

	// Rule 3: boids try to match the speed of surrounding boids
	if (neighborCount3 != 0)
	{
		perceived_velocity /= neighborCount3;
		v3 = perceived_velocity*rule3Scale;
	}

	return v1 + v2 + v3;
}

/*
* Implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}

	// Compute a new velocity based on pos and vel1
	glm::vec3 newVel = vel1[index] + computeVelocityChange(N, index, pos, vel1);
	// Clamp the speed
	if (glm::length(newVel) > maxSpeed)
	{
		newVel = glm::normalize(newVel) * maxSpeed;
	}
	// Record the new velocity into vel2. 
	// Question: why NOT vel1?
	// Answer: vel1 is being read from as well in this kernel. And so if we wrote into it some threads might read 
	// in the incorrect data or they simply may not be able to read it because it is being written into, etc. This
	// is why we ping-pog the velocity buffers
	vel2[index] = newVel;
}

/*
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel)
{
	// Update position by velocity
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
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

// Method of computing a 1D index from a 3D grid index.
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution)
{
	return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
	glm::vec3 gridMin, float inverseCellWidth,
	glm::vec3 *pos, int *indices, int *gridIndices)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}

	// Go through the boids and determine which grid cell to bin them into
	glm::ivec3 boidPos = (pos[index] - gridMin) * inverseCellWidth;
	gridIndices[index] = gridIndex3Dto1D(boidPos.x, boidPos.y, boidPos.z, gridResolution);
	// Set up a parallel array of integer indices as pointers to the actual
	// boid data in pos and vel1/vel2
	indices[index] = index;
}

// This is useful for indicating that a cell does not enclose any boids
// Called at the beginning of every step of a simulation to reset the buffer values to a default value which 
// tells us if the cell holds any boids or not
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N)
	{
		intBuffer[index] = value;
	}
}

// Identify the start and end points of each gridcell in the gridIndices array.
__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
	int *gridCellStartIndices, int *gridCellEndIndices)
{
	//go through particleGridIndices identifying when there is a change in there value, 
	//which signifies a change in the gridcell we are dealing with
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}

	if (index == 0) //edge case
	{
		gridCellStartIndices[particleGridIndices[index]] = 0;
	}
	else if (index == N - 1) //edge case
	{
		gridCellEndIndices[particleGridIndices[index]] = N - 1;
	}
	else if (particleGridIndices[index] != particleGridIndices[index + 1])
	{
		//inbetween grid cells with no boids are set to -1  --> done before when both the arrays were reset to -1

		//change in gridcell
		gridCellEndIndices[particleGridIndices[index]] = index;
		gridCellStartIndices[particleGridIndices[index + 1]] = index + 1;
	}
}

// Store the reshuffled position and velocity buffers that are more memory coherent in new coherentPos and coherentVel buffers 
__global__ void kernSetCoherentPosVel(int N, int *particleArrayIndices,
	int *gridCellStartIndices, int *gridCellEndIndices,
	const glm::vec3 *pos, const glm::vec3 *vel,
	glm::vec3 *coherentPos, glm::vec3 *coherentVel)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}

	int coherentindex = particleArrayIndices[index];

	coherentPos[index] = pos[coherentindex];
	coherentVel[index] = vel[coherentindex];
}

__global__ void kernUpdateVelNeighborSearchScattered(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int *gridCellStartIndices, int *gridCellEndIndices,
	int *particleArrayIndices,
	glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2)
{
	// Update a boid's velocity using the uniform grid to reduce
	// the number of boids that need to be checked.

	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}

	//find boid position
	//then use that position to determine the grid cell the boid belongs to
	//use that information to find the 8 cells you have to check
	glm::ivec3 boidPos = (pos[index] - gridMin) * inverseCellWidth;
	int x = boidPos.x;
	int y = boidPos.y;
	int z = boidPos.z;

	glm::vec3 v1 = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 v2 = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 v3 = glm::vec3(0.0f, 0.0f, 0.0f);

	glm::vec3 percieved_center_of_mass = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 perceived_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 separate_vector = glm::vec3(0.0f, 0.0f, 0.0f);

	int neighborCount1 = 0;
	int neighborCount3 = 0;

	float distance = 0.0f;

	for (int i = -1; i < 1; i++)
	{
		for (int j = -1; j < 1; j++)
		{
			for (int k = -1; k < 1; k++)
			{
				int _x = x + i;
				int _y = y + j;
				int _z = z + k;

				_x = imax(_x, 0);
				_y = imax(_y, 0);
				_z = imax(_z, 0);

				_x = imin(_x, gridResolution - 1);
				_y = imin(_y, gridResolution - 1);
				_z = imin(_z, gridResolution - 1);

				int boidGridCellindex = gridIndex3Dto1D(_x, _y, _z, gridResolution);

				// Identify which cells may contain neighboring boids. This isn't always 8.
				if (gridCellStartIndices[boidGridCellindex] != -1)
				{
					//we know the grid cell is empty if its start or end indices have been set to -1

					// For each cell that contains boids and needs to be checked, 
					// read the start/end indices in the boid pointer array.

					// Now go through the boids in that grid cell and apply the rules 
					// to it if it falls within the neighbor hood distance
					for (int h = gridCellStartIndices[boidGridCellindex]; h <= gridCellEndIndices[boidGridCellindex]; h++)
					{
						//Access each boid in the cell and compute velocity change from
						int bindex = particleArrayIndices[h];
						if (h != index)
						{
							//Compute velocity change based on rules
							distance = glm::distance(pos[bindex], pos[index]);
							if (distance < rule1Distance)
							{
								percieved_center_of_mass += pos[bindex];
								neighborCount1++;
							}

							if (distance < rule2Distance)
							{
								separate_vector -= (pos[bindex] - pos[index]);
							}

							if (distance < rule3Distance)
							{
								perceived_velocity += vel1[bindex];
								neighborCount3++;
							}
						}
					}
				}
			}
		}
	}

	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	if (neighborCount1 != 0)
	{
		percieved_center_of_mass /= neighborCount1;
		v1 = (percieved_center_of_mass - pos[index])*rule1Scale;
	}

	// Rule 2: boids try to stay a distance d away from each other
	v2 = separate_vector*rule2Scale;

	// Rule 3: boids try to match the speed of surrounding boids
	if (neighborCount3 != 0)
	{
		perceived_velocity /= neighborCount3;
		v3 = perceived_velocity*rule3Scale;
	}

	glm::vec3 newVel = vel1[index] + v1 + v2 + v3;

	// Clamp the speed change before putting the new speed in vel2
	if (glm::length(newVel) > maxSpeed)
	{
		newVel = glm::normalize(newVel) * maxSpeed;
	}
	vel2[index] = newVel;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int *gridCellStartIndices, int *gridCellEndIndices,
	glm::vec3 *coherentPos, glm::vec3 *coherentVel, glm::vec3 *vel2)
{
	// Very similar to kernUpdateVelNeighborSearchScattered, except with one less level of indirection.
	// This should expect gridCellStartIndices and gridCellEndIndices to refer directly to coherentPos and CoherentVel.	

	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}

	//find boid position
	//then use that position to determine the gridcell it belongs to
	//use that information to find the 8 cells you have to check
	glm::ivec3 boidPos = (coherentPos[index] - gridMin) * inverseCellWidth;
	int x = boidPos.x;
	int y = boidPos.y;
	int z = boidPos.z;

	glm::vec3 v1 = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 v2 = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 v3 = glm::vec3(0.0f, 0.0f, 0.0f);

	glm::vec3 percieved_center_of_mass = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 perceived_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 separate_vector = glm::vec3(0.0f, 0.0f, 0.0f);

	int neighborCount1 = 0;
	int neighborCount3 = 0;

	float distance = 0.0f;

	// DIFFERENCE: For best results, consider what order the cells should be
	// checked in to maximize the memory benefits of reordering the boids data.
	for (int k = -1; k < 1; k++) //z axis
	{
		for (int j = -1; j < 1; j++) //y axis
		{
			for (int i = -1; i < 1; i++) //x axis
			{
				int _x = x + i;
				int _y = y + j;
				int _z = z + k;

				_x = imax(_x, 0);
				_y = imax(_y, 0);
				_z = imax(_z, 0);

				_x = imin(_x, gridResolution - 1);
				_y = imin(_y, gridResolution - 1);
				_z = imin(_z, gridResolution - 1);

				int boidGridCellindex = gridIndex3Dto1D(_x, _y, _z, gridResolution);

				// Identify which cells may contain neighbors. This isn't always 8.
				// For each cell, read the start/end indices in the boid pointer array.

				if (gridCellStartIndices[boidGridCellindex] != -1)
				{
					//we know the grid cell is empty if its start or end indices have been set to -1

					//now go through the boids in that grid cell and apply the rules 
					//to it if it falls within the neighbor hood distance
					for (int h = gridCellStartIndices[boidGridCellindex]; h <= gridCellEndIndices[boidGridCellindex]; h++)
					{
						if (h != index)
						{
							// Access each boid in the cell and compute velocity change from
							// the boids rules, if this boid is within the neighborhood distance.
							distance = glm::distance(coherentPos[h], coherentPos[index]);
							if (distance < rule1Distance)
							{
								percieved_center_of_mass += coherentPos[h];
								neighborCount1++;
							}

							if (distance < rule2Distance)
							{
								separate_vector -= (coherentPos[h] - coherentPos[index]);
							}

							if (distance < rule3Distance)
							{
								perceived_velocity += coherentVel[h];
								neighborCount3++;
							}
						}
					}
				}
			}
		}
	}

	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	if (neighborCount1 != 0)
	{
		percieved_center_of_mass /= neighborCount1;
		v1 = (percieved_center_of_mass - coherentPos[index])*rule1Scale;
	}

	// Rule 2: boids try to stay a distance d away from each other
	v2 = separate_vector*rule2Scale;

	// Rule 3: boids try to match the speed of surrounding boids
	if (neighborCount3 != 0)
	{
		perceived_velocity /= neighborCount3;
		v3 = perceived_velocity*rule3Scale;
	}

	glm::vec3 newVel = coherentVel[index] + v1 + v2 + v3;

	// Clamp the speed change before putting the new speed in vel2
	if (glm::length(newVel) > maxSpeed)
	{
		newVel = glm::normalize(newVel) * maxSpeed;
	}
	vel2[index] = newVel;
}


//Step the entire N-body simulation by `dt` seconds.
void Boids::stepSimulationNaive(float dt)
{
	//Step the simulation forward in time.
	//Setup thread/block execution configuration
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);//no dim1,  dim3 automatically makes the ther dimensions 0

																	 //update boid velocities
	kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

	//update boid positions
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel1);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	//Ping-pong the velocity buffers
	std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt)
{
	// Uniform Grid Neighbor search using Thrust sort.

	dim3 fullBlocksPerGrid_gridsize((gridCellCount + blockSize - 1) / blockSize);//no dim1,  dim3 automatically makes the ther dimensions 0
	dim3 fullBlocksPerGrid_boids((numObjects + blockSize - 1) / blockSize);

	// Reset buffers start and end indices buffers
	kernResetIntBuffer << <fullBlocksPerGrid_gridsize, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <fullBlocksPerGrid_gridsize, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

	// Label each particle with its array index as well as its grid index.
	// Use 2x width grids.
	// recompute grid cell indices and particlearray indices every timestep
	kernComputeIndices << <fullBlocksPerGrid_boids, blockSize >> > (numObjects, gridSideCount,
		gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

	// Now sort the dev_particleGridIndices so that boids belonging to the same grid cell
	// are next to each other in the gridIndices array --> Use Thrust to sort the array

	// Wrap device vectors in thrust iterators for use with thrust.
	thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);
	thrust::device_ptr<int> dev_thrust_values(dev_particleArrayIndices);
	// Unstable key sort using Thrust. A stable sort isn't necessary
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, dev_thrust_values);
	checkCUDAErrorWithLine("thrust sorting failed!");

	// assuming the boidGridIndices are sorted, assign values to the arrays keeping 
	// track of the data in dev_particleArrayIndices for each cell.

	// unroll the loop for finding the start and end indices of each
	// cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd << <fullBlocksPerGrid_boids, blockSize >> > (numObjects, dev_particleGridIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices);

	// Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid_boids, blockSize >> > (numObjects, gridSideCount, gridMinimum,
		gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

	// Update positions
	kernUpdatePos << <fullBlocksPerGrid_boids, blockSize >> >(numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	//Ping-pong the velocity buffers
	std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationCoherentGrid(float dt)
{
	// Uniform Grid Neighbor search using Thrust sort with cell-coherent data.

	dim3 fullBlocksPerGrid_gridsize((gridCellCount + blockSize - 1) / blockSize);//no dim1,  dim3 automatically makes the ther dimensions 0
	dim3 fullBlocksPerGrid_boids((numObjects + blockSize - 1) / blockSize);

	//Reset buffers start and end indices buffers
	kernResetIntBuffer << <fullBlocksPerGrid_gridsize, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <fullBlocksPerGrid_gridsize, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
	// Label each particle with its array index as well as its grid index.
	// Use 2x width grids
	// recompute grid cell indices and particlearray indices every timestep
	kernComputeIndices << <fullBlocksPerGrid_boids, blockSize >> > (numObjects, gridSideCount,
		gridMinimum, gridInverseCellWidth,
		dev_pos, dev_particleArrayIndices,
		dev_particleGridIndices);
	// Now sort the dev_particleGridIndices so that boids belonging to the same grid cell
	// are next to each other in the gridIndices array --> Use Thrust to sort the array
	// Wrap device vectors in thrust iterators for use with thrust.
	thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);
	thrust::device_ptr<int> dev_thrust_values(dev_particleArrayIndices);
	// Unstable key sort using Thrust. A stable sort isn't necessary
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, dev_thrust_values);
	checkCUDAErrorWithLine("thrust sorting failed!");
	// Assuming the boidGridIndices are sorted, assign values to the arrays keeping 
	// track of the data in dev_particleArrayIndices for each cell.
	// Unroll the loop for finding the start and end indices of each
	// cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd << <fullBlocksPerGrid_boids, blockSize >> > (numObjects, dev_particleGridIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices);
	// BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all the boid data(position and velocity)
	// in the simulation array, such that it is memory coherent arranged in order of grid cells
	kernSetCoherentPosVel << <fullBlocksPerGrid_boids, blockSize >> > (numObjects, dev_particleArrayIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_pos, dev_vel1, dev_coherentPos, dev_coherentVel);
	// Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid_boids, blockSize >> > (numObjects, gridSideCount, gridMinimum,
		gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_coherentPos, dev_coherentVel, dev_vel2);

	// Update positions
	kernUpdatePos << <fullBlocksPerGrid_boids, blockSize >> >(numObjects, dt, dev_coherentPos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdatePos failed!");
	// Ping-pong Coherent and regular pos buffers
	std::swap(dev_coherentPos, dev_pos);
	// Ping-pong the velocity buffers
	std::swap(dev_vel1, dev_vel2);
}
//Free memory that was allocated in initSimulation
void Boids::endSimulation()
{
	//Free any buffers here
	cudaFree(dev_vel1);
	cudaFree(dev_vel2);
	cudaFree(dev_pos);
	cudaFree(dev_particleArrayIndices);
	cudaFree(dev_particleGridIndices);
	cudaFree(dev_gridCellStartIndices);
	cudaFree(dev_gridCellEndIndices);
	cudaFree(dev_coherentPos);
	cudaFree(dev_coherentVel);
}
void Boids::unitTest()
{
	// Test unstable sort
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
	for (int i = 0; i < N; i++)
	{
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}
	// Copy data to the GPU
	cudaMemcpy(dev_intKeys, intKeys, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_intValues, intValues, sizeof(int) * N, cudaMemcpyHostToDevice);
	// Wrap device vectors in thrust iterators for use with thrust.
	thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
	thrust::device_ptr<int> dev_thrust_values(dev_intValues);
	// Example for using thrust::sort_by_key
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);
	// Copy data back to the CPU side from the GPU
	cudaMemcpy(intKeys, dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(intValues, dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");
	std::cout << "after unstable sort: " << std::endl;
	for (int i = 0; i < N; i++)
	{
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}
	// Cleanup
	delete[] intKeys;
	delete[] intValues;
	cudaFree(dev_intKeys);
	cudaFree(dev_intValues);
	checkCUDAErrorWithLine("cudaFree failed!");
	return;
}