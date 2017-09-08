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

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3 *dev_vel1_shuffled;
glm::vec3 *dev_pos_shuffled;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount; //how many cells are there in total
int gridSideCount; //how many cells on one side of unit grid
float gridCellWidth; //side length of each grid cell
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
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;//Question: why halfSideCount?
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;
  std::cout << "gridMinimum is: " << gridMinimum.x << " " << gridMinimum.y << " " << gridMinimum.z << std::endl;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");
  cudaThreadSynchronize();

  cudaMalloc((void**)&dev_vel1_shuffled, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1_shuffled failed!");

  cudaMalloc((void**)&dev_pos_shuffled, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_shuffled failed!");
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
	glm::vec3 velocityChange = glm::vec3(0);
	glm::vec3 perceived_center = glm::vec3(0);
	int count = 0;
	for (int i = 0; i < N; i++) {
		if (i != iSelf && glm::length(pos[i] - pos[iSelf]) < rule1Distance) {
			perceived_center += pos[i];
			count++;
		}
	}
	if (count > 0) {
		perceived_center /= count;
		velocityChange += (perceived_center - pos[iSelf]) * rule1Scale;
	}

  // Rule 2: boids try to stay a distance d away from each other
	glm::vec3 c(0,0,0);
	for (int i = 0; i < N; i++) {
		if (i != iSelf && glm::length(pos[i] - pos[iSelf]) < rule2Distance) {
			c -= (pos[i] - pos[iSelf]);
		}
	}
	velocityChange += c*rule2Scale;
  // Rule 3: boids try to match the speed of surrounding boids
	count = 0;
	glm::vec3 perceived_velocity = glm::vec3(0.f);
	for (int i = 0; i < N; i++) {
		if (pos[i] != pos[iSelf] && glm::length(pos[i] - pos[iSelf]) < rule3Distance){
			perceived_velocity += vel[i];
			count++;
		}
	}
	if (count > 0) {
		perceived_velocity /= count;
	}	
	velocityChange += perceived_velocity*rule3Scale;
	return vel[iSelf]+velocityChange;
  //return glm::vec3(0.0f, 0.0f, 0.0f);
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	glm::vec3 newVel = computeVelocityChange(N, index, pos, vel1);
	glm::vec3 newVelFinal = newVel;
	
  // Clamp the speed
	if (glm::length(newVel) > maxSpeed) {
		newVelFinal = glm::normalize(newVel);
	}
  // Record the new velocity into vel2. Question: why NOT vel1?
	//Because we still need current state vel1 to determine other boid velocity change
	vel2[index] = newVelFinal;
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
	if (x < 0 || x >= gridResolution) {
		return -1;
	}
	if (y < 0 || y >= gridResolution) {
		return -1;
	}
	if (z < 0 || z >= gridResolution) {
		return -1;
	}
	return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int numObjects, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
	int index = threadIdx.x + (blockIdx.x * blockDim.x); //get the parallel thread Id
	if (index >= numObjects) {
		return;
	}
	glm::vec3 thisPos = pos[index];
	int xIndex = (thisPos.x-gridMin.x)*inverseCellWidth;
	int yIndex = (thisPos.y-gridMin.y)*inverseCellWidth;
	int zIndex = (thisPos.z-gridMin.z)*inverseCellWidth;
	gridIndices[index] = gridIndex3Dto1D(xIndex, yIndex, zIndex, gridResolution);

    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
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

__global__ void kernIdentifyCellStartEnd(int cellNum, int objNum, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.	
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
	int cellIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (cellIndex >= cellNum) {
		return;
	}
	gridCellStartIndices[cellIndex] = -1;
	gridCellEndIndices[cellIndex] = -1;
	bool foundCellIndex = false;
	int i = 0;
	for (i = 0; i < objNum; i++) {		
		if (particleGridIndices[i] == cellIndex && !foundCellIndex) {
			gridCellStartIndices[cellIndex] = i;
			foundCellIndex = true;
		}
		if (particleGridIndices[i] != cellIndex && foundCellIndex) {
			gridCellEndIndices[cellIndex] = i-1;
			break;
		}
		if (particleGridIndices[i] > cellIndex) {
			//Sorted particleGridIndices, so if it is bigger, then it means we are not gonna find it
			break;
		}
	}
	if (i == objNum && foundCellIndex) {
		gridCellEndIndices[cellIndex] = i - 1;
	}
}

//Added by Yuxin, may not be correct due to the division number should be all neighboring particles instead of particles in one neighboring cell
__device__ glm::vec3 computeVelocityChangeNeighborSearchScattered(int iSelf, int neighborCellIndex, const glm::vec3 *pos, const glm::vec3 *vel,
	const int *gridCellStartIndices, const int *gridCellEndIndices, const int*particleArrayIndices) {
	// - For each cell, read the start/end indices in the boid pointer array.
	// - Access each boid in the cell and compute velocity change from
	//   the boids rules, if this boid is within the neighborhood distance.
	glm::vec3 velocityChange = glm::vec3(0);
	glm::vec3 perceived_center = glm::vec3(0);
	glm::vec3 perceived_velocity = glm::vec3(0.f);
	glm::vec3 c(0, 0, 0);
	int perceivedCenterCount = 0;
	int surroundSpeedCount = 0;
	int startIndex = gridCellStartIndices[neighborCellIndex];
	int endIndex = gridCellEndIndices[neighborCellIndex];
	if (startIndex != -1 && endIndex != -1) {
		for (int neighberIndex = startIndex; neighberIndex <= endIndex; neighberIndex++) {
			int particleIndex = particleArrayIndices[neighberIndex];
			// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
			if (particleIndex != iSelf && glm::length(pos[particleIndex] - pos[iSelf]) < rule1Distance) {
				perceived_center += pos[particleIndex];
				perceivedCenterCount++;
			}
			// Rule 2: boids try to stay a distance d away from each other			
			if (particleIndex != iSelf && glm::length(pos[particleIndex] - pos[iSelf]) < rule2Distance) {
				c -= (pos[particleIndex] - pos[iSelf]);
			}
			// Rule 3: boids try to match the speed of surrounding boids
			if (pos[particleIndex] != pos[iSelf] && glm::length(pos[particleIndex] - pos[iSelf]) < rule3Distance) {
				perceived_velocity += vel[particleIndex];
				surroundSpeedCount++;
			}
		}

		if (perceivedCenterCount > 0) {
			perceived_center /= perceivedCenterCount;
			velocityChange += (perceived_center - pos[iSelf]) * rule1Scale;
		}
		velocityChange += c*rule2Scale;
		if (surroundSpeedCount > 0) {
			perceived_velocity /= surroundSpeedCount;
		}
		velocityChange += perceived_velocity*rule3Scale;
	}
	return velocityChange;
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
	int index = threadIdx.x + (blockIdx.x * blockDim.x); //get the parallel thread Id
	if (index >= N) {
		return;
	}
	glm::vec3 thisPos = pos[index];
	int xIndex = (thisPos.x-gridMin.x)*inverseCellWidth;
	int yIndex = (thisPos.y-gridMin.y)*inverseCellWidth;
	int zIndex = (thisPos.z-gridMin.z)*inverseCellWidth;
  // - Identify which cells may contain neighbors. This isn't always 8. ??Questions which 8 cells??
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
 
	//Searching for 27 neighbors
	glm::vec3 velocityChange = glm::vec3(0);
	glm::vec3 perceived_center = glm::vec3(0);
	glm::vec3 perceived_velocity = glm::vec3(0.f);
	glm::vec3 c(0, 0, 0);
	int perceivedCenterCount = 0;
	int surroundSpeedCount = 0;
	int neighborCellIndex = -1;	
	for (int xNeighbor = xIndex - 1; xNeighbor <= xIndex + 1; xNeighbor++) {
		for (int yNeibor = yIndex - 1; yNeibor <= yIndex + 1; yNeibor++) {
			for (int zNeibor = zIndex - 1; zNeibor <= zIndex + 1; zNeibor++) {
				neighborCellIndex = gridIndex3Dto1D(xNeighbor, yNeibor, zNeibor, gridResolution);
				if (neighborCellIndex >= 0) {
					//velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, 
														//pos, vel1, gridCellStartIndices, gridCellEndIndices, particleArrayIndices);										
					int startIndex = gridCellStartIndices[neighborCellIndex];
					int endIndex = gridCellEndIndices[neighborCellIndex];
					if (startIndex != -1 && endIndex != -1) {
						for (int neighberIndex = startIndex; neighberIndex <= endIndex; neighberIndex++) {
							int particleIndex = particleArrayIndices[neighberIndex];
							// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
							if (particleIndex != index && glm::length(pos[particleIndex] - thisPos) < rule1Distance) {
								perceived_center += pos[particleIndex];
								perceivedCenterCount++;
							}
							// Rule 2: boids try to stay a distance d away from each other			
							if (particleIndex != index && glm::length(pos[particleIndex] - thisPos) < rule2Distance) {
								c -= (pos[particleIndex] - thisPos);
							}
							// Rule 3: boids try to match the speed of surrounding boids
							if (particleIndex != index && glm::length(pos[particleIndex] - thisPos) < rule3Distance) {
								perceived_velocity += vel1[particleIndex];
								surroundSpeedCount++;
							}
						}
					}
				}				
			}
		}
	}
	if (perceivedCenterCount > 0) {
		perceived_center /= perceivedCenterCount;
		velocityChange += (perceived_center - thisPos) * rule1Scale;
	}
	velocityChange += c*rule2Scale;
	if (surroundSpeedCount > 0) {
		perceived_velocity /= surroundSpeedCount;
	}
	velocityChange += perceived_velocity*rule3Scale;

	/* searching for 8 neighbors, question: how to implement???
	float xPortion = (thisPos.x - cellWidth*xIndex) / cellWidth;
	float yPortion = (thisPos.y - cellWidth*yIndex) / cellWidth;
	float zPortion = (thisPos.z - cellWidth*zIndex) / cellWidth;
	bool leftX = xPortion < 0.5 ? true : false;
	bool lowerY = yPortion < 0.5 ? true : false;
	bool frontZ = zPortion < 0.5 ? true : false;
	if (leftX) {	
		//left X
		if (xIndex > 0) {
			neighborCellIndex = gridIndex3Dto1D(xIndex - 1, yIndex, zIndex, gridResolution);
			velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1, gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
			if (lowerY) {
				//left X + lower Y
				if (yIndex > 0) {
					neighborCellIndex = gridIndex3Dto1D(xIndex - 1, yIndex-1, zIndex, gridResolution);
					velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1, gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
					neighborCellIndex = gridIndex3Dto1D(xIndex, yIndex - 1, zIndex, gridResolution);
					velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1, gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
					if (frontZ) {
						//left X + lower Y + front Z
						if (zIndex > 0) {
							neighborCellIndex = gridIndex3Dto1D(xIndex, yIndex, zIndex-1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1, 
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
							neighborCellIndex = gridIndex3Dto1D(xIndex-1, yIndex - 1, zIndex-1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1, 
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
							neighborCellIndex = gridIndex3Dto1D(xIndex, yIndex - 1, zIndex-1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
							neighborCellIndex = gridIndex3Dto1D(xIndex-1, yIndex, zIndex - 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
						}
					}
					else {
						//left X + lower Y + back Z
						if (zIndex < gridSideCount - 1) {
							neighborCellIndex = gridIndex3Dto1D(xIndex - 1, yIndex - 1, zIndex + 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
							neighborCellIndex = gridIndex3Dto1D(xIndex - 1, yIndex, zIndex + 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
							neighborCellIndex = gridIndex3Dto1D(xIndex, yIndex - 1, zIndex + 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
							neighborCellIndex = gridIndex3Dto1D(xIndex, yIndex, zIndex + 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
						}

					}
				}
			}
			else { 
				//left X + upper Y
				if (yIndex < gridSideCount - 1) {
					neighborCellIndex = gridIndex3Dto1D(xIndex - 1, yIndex + 1, zIndex, gridResolution);
					velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1, gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
					neighborCellIndex = gridIndex3Dto1D(xIndex, yIndex + 1, zIndex, gridResolution);
					velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1, gridCellStartIndices, gridCellEndIndices, particleArrayIndices);

					if (frontZ) {
						//left X + upper Y + front Z
						if (zIndex > 0) {
							neighborCellIndex = gridIndex3Dto1D(xIndex - 1, yIndex + 1, zIndex - 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
							neighborCellIndex = gridIndex3Dto1D(xIndex, yIndex + 1, zIndex - 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
							neighborCellIndex = gridIndex3Dto1D(xIndex-1, yIndex, zIndex - 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
							neighborCellIndex = gridIndex3Dto1D(xIndex, yIndex, zIndex - 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
						}
					}
					else {
						//left X + upper Y + back Z
						if (zIndex < gridSideCount - 1) {
							neighborCellIndex = gridIndex3Dto1D(xIndex - 1, yIndex + 1, zIndex + 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
							neighborCellIndex = gridIndex3Dto1D(xIndex, yIndex + 1, zIndex + 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
							neighborCellIndex = gridIndex3Dto1D(xIndex - 1, yIndex, zIndex + 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
							neighborCellIndex = gridIndex3Dto1D(xIndex, yIndex, zIndex + 1, gridResolution);
							velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, pos, vel1,
								gridCellStartIndices, gridCellEndIndices, particleArrayIndices);
						}

					}
				}
			}
		}
	}*/
  
	// - Clamp the speed change before putting the new speed in vel2
	//check which portion of the cell is the particle in, there are 8 portions each cell
	//on xdirection
	glm::vec3 newVel = vel1[index] + velocityChange;
	glm::vec3 finalNewVel = newVel;
	if (glm::length(newVel) > maxSpeed) {
		finalNewVel = glm::normalize(newVel);
	}
	vel2[index] = finalNewVel;
}

//Added by Yuxin, rearrange the position and velocity data to match sorted particle cell indice
__global__ void kernShufflePosVel(int N, int *particleArrayIndices, glm::vec3 *pos, glm::vec3 *pos_shuffled, 
	glm::vec3 *vel1, glm::vec3 *vel1_shuffled) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x); //get the parallel thread Id
	if (index >= N) {
		return;
	}
	int particleIndex_shuffled = particleArrayIndices[index];
	pos_shuffled[index] = pos[particleIndex_shuffled];
	vel1_shuffled[index] = vel1[particleIndex_shuffled];
}


__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos_shuffled, glm::vec3 *vel1_shuffled, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
	int index = threadIdx.x + (blockIdx.x * blockDim.x); //get the parallel thread Id
	if (index >= N) {
		return;
	}
	glm::vec3 thisPos = pos_shuffled[index];
	int xIndex = (thisPos.x - gridMin.x)*inverseCellWidth;
	int yIndex = (thisPos.y - gridMin.y)*inverseCellWidth;
	int zIndex = (thisPos.z - gridMin.z)*inverseCellWidth;
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  //Searching for 27 neighbors
	glm::vec3 velocityChange = glm::vec3(0);
	glm::vec3 perceived_center = glm::vec3(0);
	glm::vec3 perceived_velocity = glm::vec3(0.f);
	glm::vec3 c(0, 0, 0);
	int perceivedCenterCount = 0;
	int surroundSpeedCount = 0;
	int neighborCellIndex = -1;
	for (int xNeighbor = xIndex - 1; xNeighbor <= xIndex + 1; xNeighbor++) {
		for (int yNeibor = yIndex - 1; yNeibor <= yIndex + 1; yNeibor++) {
			for (int zNeibor = zIndex - 1; zNeibor <= zIndex + 1; zNeibor++) {
				neighborCellIndex = gridIndex3Dto1D(xNeighbor, yNeibor, zNeibor, gridResolution);
				if (neighborCellIndex >= 0) {
					//velocityChange += computeVelocityChangeNeighborSearchScattered(index, neighborCellIndex, 
					//pos, vel1, gridCellStartIndices, gridCellEndIndices, particleArrayIndices);										
					int startIndex = gridCellStartIndices[neighborCellIndex];
					int endIndex = gridCellEndIndices[neighborCellIndex];
					if (startIndex != -1 && endIndex != -1) {
						for (int neighberIndex = startIndex; neighberIndex <= endIndex; neighberIndex++) {
							// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
							if (neighberIndex != index && glm::length(pos_shuffled[neighberIndex] - thisPos) < rule1Distance) {
								perceived_center += pos_shuffled[neighberIndex];
								perceivedCenterCount++;
							}
							// Rule 2: boids try to stay a distance d away from each other			
							if (neighberIndex != index && glm::length(pos_shuffled[neighberIndex] - thisPos) < rule2Distance) {
								c -= (pos_shuffled[neighberIndex] - thisPos);
							}
							// Rule 3: boids try to match the speed of surrounding boids
							if (neighberIndex != index &&
								glm::length(pos_shuffled[neighberIndex] - thisPos) < rule3Distance) {
								perceived_velocity += vel1_shuffled[neighberIndex];
								surroundSpeedCount++;
							}
						}
					}
				}
			}
		}
	}
	if (perceivedCenterCount > 0) {
		perceived_center /= perceivedCenterCount;
		velocityChange += (perceived_center - pos_shuffled[index]) * rule1Scale;
	}
	velocityChange += c*rule2Scale;
	if (surroundSpeedCount > 0) {
		perceived_velocity /= surroundSpeedCount;
	}
	velocityChange += perceived_velocity*rule3Scale;

  // - Clamp the speed change before putting the new speed in vel2
	glm::vec3 newVel = vel1_shuffled[index] + velocityChange;
	glm::vec3 finalNewVel = newVel;
	if (glm::length(newVel) > maxSpeed) {
		finalNewVel = glm::normalize(newVel);
	}
	vel2[index] = finalNewVel;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	// TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernUpdateVelocityBruteForce <<<fullBlocksPerGrid, blockSize >>>(numObjects, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdatePos failed!");
  // TODO-1.2 ping-pong the velocity buffers
	cudaMemcpy(dev_vel1, dev_vel2, numObjects* sizeof(glm::vec3),cudaMemcpyDeviceToDevice);

}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
	thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);
	thrust::device_ptr<int> dev_thrust_values(dev_particleArrayIndices);
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, dev_thrust_values);
	//Debug
	/*int *particleGridIndex = new int[numObjects];
	int *particleIndex = new int[numObjects];
	// How to copy data back to the CPU side from the GPU
	cudaMemcpy(particleGridIndex, dev_particleGridIndices, sizeof(int) * numObjects, cudaMemcpyDeviceToHost);
	cudaMemcpy(particleIndex, dev_particleArrayIndices, sizeof(int) * numObjects, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");

	std::cout << "after unstable sort: " << std::endl;
	for (int i = 0; i < 100; i++) {
		std::cout << "  key: " << particleGridIndex[i];
		std::cout << " value: " << particleIndex[i] << std::endl;
	}

	// cleanup
	delete[] particleGridIndex;
	delete[] particleIndex;*/
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
	dim3 fullCellsPerGrid((gridCellCount + blockSize - 1) / blockSize);
	kernIdentifyCellStartEnd << <fullCellsPerGrid, blockSize >> >(gridCellCount, numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  // - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> >
	(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
  // - Update positions
	kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
  // - Ping-pong buffers as needed
	cudaMemcpy(dev_vel1, dev_vel2, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:  

  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);
	thrust::device_ptr<int> dev_thrust_values(dev_particleArrayIndices);
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, dev_thrust_values);

	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	dim3 fullCellsPerGrid((gridCellCount + blockSize - 1) / blockSize);
	kernIdentifyCellStartEnd << <fullCellsPerGrid, blockSize >> >(gridCellCount, numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

	// - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
	//   the particle data in the simulation array.
	//   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
	kernShufflePosVel << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleArrayIndices, dev_pos, dev_pos_shuffled,
		dev_vel1, dev_vel1_shuffled);
	
		// - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> >
		(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
			dev_pos_shuffled, dev_vel1_shuffled, dev_vel2);
	// - Update positions
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos_shuffled, dev_vel2);
	// - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

	/*glm::vec3* temp;
	temp = dev_pos;
	dev_pos = dev_pos_shuffled;
	dev_pos_shuffled = temp;

	temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = dev_vel1;*/

	cudaMemcpy(dev_pos, dev_pos_shuffled, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_vel1, dev_vel2, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

	//Debug
	/*glm::vec3 *particlePos = new glm::vec3[numObjects];
	glm::vec3 *particleVel = new glm::vec3[numObjects];
	// How to copy data back to the CPU side from the GPU
	cudaMemcpy(particlePos, dev_pos, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToHost);
	cudaMemcpy(particleVel, dev_vel1, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");

	std::cout << "after update vel and pos in coherent grids: " << std::endl;
	for (int i = 0; i < 100; i++) {
		std::cout << " position: " << "["<< particlePos[i].x<<" "<< particlePos[i].y<<" "<< particlePos[i].z<<std::endl;
		std::cout << " velocity: " << "[" << particleVel[i].x << " " << particleVel[i].y << " " << particleVel[i].z << std::endl;
	}

	// cleanup
	delete[] particlePos;
	delete[] particleVel;*/
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

  cudaFree(dev_vel1_shuffled);
  cudaFree(dev_pos_shuffled);
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
