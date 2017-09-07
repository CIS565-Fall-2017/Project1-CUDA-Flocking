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
// Consider why you would need two velocity buffers in a simulation where each. To maintain state for this frame
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // tells you where to go look in dev_pos and dev_velX for this particles info
int *dev_particleGridIndices; // What grid cell is this particle in
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell? if -1 then no particles in this cell

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
// Don't forget to free in Boids::endSimulation
glm::vec3* dev_cohpos;
glm::vec3* dev_cohvel1;
glm::vec3* dev_cohvel2;


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
  a = (a + 0x165667b1) + (a << 5 );
  a = (a + 0xd3a2646c) ^ (a << 9 );
  a = (a + 0xfd7046c5) + (a << 3 );
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

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here. Don't forget to free in Boids::endSimulation
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_cohpos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_cohpos failed!");

  cudaMalloc((void**)&dev_cohvel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_cohvel1 failed!");

  cudaMalloc((void**)&dev_cohvel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_cohvel2 failed!");

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
__device__ glm::vec3 computeVelocityChange(const int N, const int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
	//rule1
	glm::vec3 rule1vel(0.f);
	int countrule1 = 0;

	//rule2
	glm::vec3 rule2vel(0.f);

	//rule3
	glm::vec3 rule3vel(0.f);
	int countrule3 = 0;

	const glm::vec3 iselfpos = pos[iSelf];

	for (int i = 0; i < N; ++i) {
		if (i == iSelf) { continue; }
		const glm::vec3 displacement = pos[i] - iselfpos;
		const float distance = glm::length(displacement);

		if (distance < rule1Distance) {// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
			rule1vel += pos[i];
			countrule1++;
		}

		if (distance < rule2Distance) {// Rule 2: boids try to stay a distance d away from each other
			rule2vel -= displacement;
		}

		if (distance < rule3Distance) {// Rule 3: boids try to match the speed of surrounding boids
			rule3vel += vel[i];
			countrule3++;
		}
	}

	//rule1
	if (countrule1 > 0) {
		rule1vel = rule1Scale * ( (rule1vel / (float)countrule1) - iselfpos ); //scaled vector to center of mass
	}

	//rule2
	rule2vel = rule2Scale * rule2vel; //pushed away from near neighbors

	//rule3
	if (countrule3 > 0) {
		rule3vel = rule3Scale * (rule3vel / (float)countrule3);//scaled ave velocity  
	}

	return vel[iSelf] + rule1vel + rule2vel + rule3vel;
	//return glm::vec3(0);
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(const int N, const glm::vec3 *pos,
  const glm::vec3 *vel1, glm::vec3 *vel2) {
  const int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) { return; }

  // Compute a new velocity based on pos and vel1
  glm::vec3 newvel = computeVelocityChange(N, index, pos, vel1);

  // Clamp the speed
  const float newvelspeed = glm::length(newvel);
  //newvel *= (1.f / newvelspeed) * glm::min(maxSpeed, newvelspeed);//normalizes then scales, NTS: branch still hidden inside glm::min, might as well expose the thread divergence
  if (newvelspeed > maxSpeed) {
	  newvel *= (maxSpeed / newvelspeed);//normalizes then scales
  }


  // Record the new velocity into vel2. Question: why NOT vel1?
  vel2[index] = newvel;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) { return; }

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

__global__ void kernUpdatePosCoh(const int N, const float dt, 
	const glm::vec3* cohpos, const glm::vec3* cohvel2, glm::vec3* regularpos, const int* particleArrayIndices) {
  const int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) { return; }


  glm::vec3 thisPos = cohpos[index];
  thisPos += cohvel2[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  const int regularindex = particleArrayIndices[index];
  regularpos[regularindex] = thisPos;

}
// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(const glm::ivec3 cellxyz, const int gridResolution) {
  return cellxyz.x + cellxyz.y * gridResolution + cellxyz.z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(const int N, const int gridResolution,
  const glm::vec3 gridMin, const float inverseCellWidth,
  const glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N) { return; }

	glm::ivec3 cellxyz = (pos[index] - gridMin) * inverseCellWidth;//need to shift the points by the opposite of the gridMin vector so that points in the gridMin cell get 0,0,0 indices
	gridIndices[index] = gridIndex3Dto1D(cellxyz, gridResolution);
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
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N) { return; }
	const int previndex = index - 1;
	if(previndex < 0 || particleGridIndices[previndex] != particleGridIndices[index]) { 
		const int gridcell = particleGridIndices[index];
		gridCellStartIndices[gridcell] = index;
		if (previndex >= 0) {
			const int gridcell = particleGridIndices[previndex];
			gridCellEndIndices[gridcell] = previndex; 
		} 
	} 
}

__global__ void kernCopyPosVelToCohPosVel(const int N, const int* particleArrayIndices, 
	const glm::vec3* regularpos, const glm::vec3* regularvel1, glm::vec3* cohpos, glm::vec3* cohvel1) {
	//copies the normal pos and vel1 data to cell aligned pos and vel slots
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N) { return; }

	const int regularindex = particleArrayIndices[index];
	cohpos[index]  = regularpos[regularindex];
	cohvel1[index] = regularvel1[regularindex];
}

__global__ void kernCopyCohVel2ToVel1(const int N, const int* particleArrayIndices, const glm::vec3* cohvel2, glm::vec3* vel1) {
	//copies the normal pos and vel1 data to cell aligned pos and vel slots
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N) { return; }

	const int regularindex = particleArrayIndices[index];
	vel1[regularindex] = cohvel2[index];
}

__global__ void kernUpdateVelNeighborSearchScattered(const int N, const int gridResolution, const glm::vec3 gridMin,
  const float inverseCellWidth, const float cellWidth,
  const int *gridCellStartIndices, const int *gridCellEndIndices,
  const int *particleArrayIndices,
  const glm::vec3 *pos, const glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N) { return; }
	glm::vec3 cellxyz_float = (pos[index] - gridMin) * inverseCellWidth; 
	glm::ivec3 cellxyz(cellxyz_float);
	const glm::vec3 iselfpos = pos[index];
	const int possible_neighbs = 8;
	int neighbs[possible_neighbs];
	int total_neighbs = 0;
  // - Identify which cells may contain neighbors. This isn't always 8.
	//determine which sub region of the cell this boid is in (upper or lower / left or right / front or back) using the cell's centroid
	//centroid is 0.5, 0.5, 0.5 in the frame of ref of the cell
	//assume the 8 blocks are for the +1, +1, +1 configuration(point sits on the positive side of xyz planes that split through the middle of the cell)
	//then use the sign bits to offset xyz indices when we iterate onto that cell
	//setup sign so that it doesnt do anything for the assumed +1, +1 +1 plane position configuration i.e. convert 1 to 0 and leave -1 as is
	//add the sign offsets to the 8 assumed neighbor offsets, if block is outside cellgrid then skip it and dont increment neighbor counter
	//in the loop bit 1 of the iterator is for left/right or x, bit 2 for forward/back or y, bit 3 is for up/down or z, assignment doesnt matter they each get hit 4 times anyway

	//glm::vec3 diff( cellxyz_float.x - cellxyz.x, cellxyz_float.y - cellxyz.y, cellxyz_float.z - cellxyz.z);
	//glm::ivec3 sign = glm::sign(diff - glm::vec3(0.5f, 0.5f, 0.5f));//1 vs -1 tells you where the point lies in relation to the xyz planes that spit the cell into 8 regions
	//sign = (sign - 1) / 2; //converts to -1 or 0, flips a border neighbor cell to the other side (-1) or leaves it there (0)
	//for (int i = 0; i < possible_neighbs; ++i) {
	//	const glm::ivec3 idx( (i&1)*sign.x + cellxyz.x, (i&2)*sign.y + cellxyz.y, (i&4)*sign.z + cellxyz.z);
 //       const int cellidx1d = gridIndex3Dto1D(idx, gridResolution);
	//	if (idx.x < 0 || idx.y < 0 || idx.z < 0 || idx.x >= gridResolution || idx.y >= gridResolution || idx.z >= gridResolution 
	//		|| gridCellStartIndices[cellidx1d] < 0) {
	//		continue;
	//	}
	//	neighbs[total_neighbs++] = cellidx1d;
	//}

	//alternate: find the min and max cells in x y z by adding the max neighborhood radius to our boid position in xyz and negative xyz
	const float maxrad = glm::max(glm::max(rule1Distance, rule2Distance), rule3Distance);
	const glm::ivec3 lowcell  = (iselfpos - glm::vec3(maxrad, maxrad, maxrad) - gridMin) * inverseCellWidth;
	const glm::ivec3 highcell = (iselfpos + glm::vec3(maxrad, maxrad, maxrad) - gridMin) * inverseCellWidth;
	for (int z = lowcell.z; z <= highcell.z; ++z) {
		if (z < 0 || z >= gridResolution) { continue; }

		for (int y = lowcell.y; y <= highcell.y; ++y) {
			if (y < 0 || y >= gridResolution) { continue; }

			for (int x = lowcell.x; x <= highcell.x; ++x) {
				if (x < 0 || x >= gridResolution) { continue; }

				neighbs[total_neighbs++] = gridIndex3Dto1D( glm::ivec3(x,y,z) , gridResolution); 

			}//x
		}//y
	}//z


	//rule1
	glm::vec3 rule1vel(0.f);
	int countrule1 = 0;
	//rule2
	glm::vec3 rule2vel(0.f);
	//rule3
	glm::vec3 rule3vel(0.f);
	int countrule3 = 0;


	// - For each cell, read the start/end indices in the boid pointer array.
	//loops over neighbs around our boid
	for (int j = 0; j < total_neighbs; ++j) {
		const int grididx1d = neighbs[j];
		const int startidx = gridCellStartIndices[grididx1d];
		const int endidx   = gridCellEndIndices[grididx1d];
		// - Access each boid in the cell and compute velocity change from
		//   the boids rules, if this boid is within the neighborhood distance.
		for (int k = startidx; k <= endidx; ++k) {
			int i = particleArrayIndices[k];

			if (i == index) { continue; }
			const glm::vec3 displacement = pos[i] - iselfpos;
			const float distance = glm::length(displacement);

			if (distance < rule1Distance) {// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
				rule1vel += pos[i];
				countrule1++;
			}

			if (distance < rule2Distance) {// Rule 2: boids try to stay a distance d away from each other
				rule2vel -= displacement;
			}

			if (distance < rule3Distance) {// Rule 3: boids try to match the speed of surrounding boids
				rule3vel += vel1[i];
				countrule3++;
			}
		} //for k
	}//for j

	//calculate the differnt rule velocities
	if (countrule1 > 0) {
		rule1vel = rule1Scale * ( (rule1vel / (float)countrule1) - iselfpos ); //scaled vector to center of mass
	}
	rule2vel = rule2Scale * rule2vel; //pushed away from near neighbors
	if (countrule3 > 0) {
		rule3vel = rule3Scale * (rule3vel / (float)countrule3);//scaled ave velocity  
	}

	// add and clamp
	glm::vec3 newvel = vel1[index] + rule1vel + rule2vel + rule3vel;
	const float newvelspeed = glm::length(newvel);
	if (newvelspeed > maxSpeed) {
		newvel *= (maxSpeed / newvelspeed);//normalizes then scales
	}
	// Record the new velocity into vel2. Question: why NOT vel1? to maintain state of this frame and not corrupt it, other threads will need the info
	vel2[index] = newvel;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  const int N, const int gridResolution, const glm::vec3 gridMin,
  const float inverseCellWidth, const float cellWidth,
  const int *gridCellStartIndices, const int *gridCellEndIndices,
  const glm::vec3 *pos, const glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N) { return; }
	const glm::vec3 iselfpos = pos[index];
	glm::vec3 cellxyz_float = (iselfpos - gridMin) * inverseCellWidth; 
	glm::ivec3 cellxyz(cellxyz_float);
	const int possible_neighbs = 8;
	int neighbs[possible_neighbs];
	int total_neighbs = 0;

	//alternate: find the min and max cells in x y z by adding the max neighborhood radius to our boid position in xyz and negative xyz
	const float maxrad = glm::max(glm::max(rule1Distance, rule2Distance), rule3Distance);
	const glm::ivec3 lowcell  = (iselfpos - glm::vec3(maxrad, maxrad, maxrad) - gridMin) * inverseCellWidth;
	const glm::ivec3 highcell = (iselfpos + glm::vec3(maxrad, maxrad, maxrad) - gridMin) * inverseCellWidth;
	for (int z = lowcell.z; z <= highcell.z; ++z) {
		if (z < 0 || z >= gridResolution) { continue; }

		for (int y = lowcell.y; y <= highcell.y; ++y) {
			if (y < 0 || y >= gridResolution) { continue; }

			for (int x = lowcell.x; x <= highcell.x; ++x) {
				if (x < 0 || x >= gridResolution) { continue; }

				neighbs[total_neighbs++] = gridIndex3Dto1D( glm::ivec3(x,y,z) , gridResolution); 

			}//x
		}//y
	}//z


	glm::vec3 rule1vel(0.f);
	int countrule1 = 0;
	glm::vec3 rule2vel(0.f);
	glm::vec3 rule3vel(0.f);
	int countrule3 = 0;


	// - For each cell, read the start/end indices in the boid pointer array.
	//loops over neighbs around our boid
	for (int j = 0; j < total_neighbs; ++j) {
		const int grididx1d = neighbs[j];
		const int startidx = gridCellStartIndices[grididx1d];
		const int endidx   = gridCellEndIndices[grididx1d];
		// - Access each boid in the cell and compute velocity change from
		//   the boids rules, if this boid is within the neighborhood distance.
		for (int i = startidx; i <= endidx; ++i) {
			//int i = particleArrayIndices[k];

			if (i == index) { continue; }
			const glm::vec3 displacement = pos[i] - iselfpos;
			const float distance = glm::length(displacement);

			if (distance < rule1Distance) {// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
				rule1vel += pos[i];
				countrule1++;
			}

			if (distance < rule2Distance) {// Rule 2: boids try to stay a distance d away from each other
				rule2vel -= displacement;
			}

			if (distance < rule3Distance) {// Rule 3: boids try to match the speed of surrounding boids
				rule3vel += vel1[i];
				countrule3++;
			}
		} //for i, boids in cell
	}//for j, cells

	//calculate the differnt rule velocities
	if (countrule1 > 0) {
		rule1vel = rule1Scale * ( (rule1vel / (float)countrule1) - iselfpos ); //scaled vector to center of mass
	}
	rule2vel = rule2Scale * rule2vel; //pushed away from near neighbors
	if (countrule3 > 0) {
		rule3vel = rule3Scale * (rule3vel / (float)countrule3);//scaled ave velocity  
	}

	// add and clamp
	glm::vec3 newvel = vel1[index] + rule1vel + rule2vel + rule3vel;
	const float newvelspeed = glm::length(newvel);
	if (newvelspeed > maxSpeed) {
		newvel *= (maxSpeed / newvelspeed);//normalizes then scales
	}
	// Record the new velocity into vel2. Question: why NOT vel1? to maintain state of this frame and not corrupt it, other threads will need the info
	vel2[index] = newvel;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) { 
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernUpdateVelocityBruteForce << < fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
	kernUpdatePos << < fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

	//no need to call cudaThreadSynchronize() because gpu executes one kernel at a time.
	//cpu dispatches these calls to the gpu and the gpu queues them while the cpu continues on its way

  // TODO-1.2 ping-pong the velocity buffers, i.e. swap the pointers
	glm::vec3* temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = temp;
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernComputeIndices <<<fullBlocksPerGrid , blockSize>>> (numObjects, gridSideCount, gridMinimum, 
		gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
	dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
	dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
	dim3 fullBlocksPerGrid_startendindices((gridCellCount + blockSize - 1) / blockSize);
	kernResetIntBuffer<< <fullBlocksPerGrid_startendindices, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer<< <fullBlocksPerGrid_startendindices, blockSize >> >(gridCellCount, dev_gridCellEndIndices  , -1);
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices);

  // - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchScattered<<< fullBlocksPerGrid, blockSize >>>(numObjects, gridSideCount, gridMinimum,
		gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

  // - Update positions
	kernUpdatePos << < fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

  // - Ping-pong buffers as needed
	glm::vec3* temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = temp;
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernComputeIndices <<<fullBlocksPerGrid , blockSize>>> (numObjects, gridSideCount, gridMinimum, 
		gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
	dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
	dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
	dim3 fullBlocksPerGrid_startendindices((gridCellCount + blockSize - 1) / blockSize);
	kernResetIntBuffer<< <fullBlocksPerGrid_startendindices, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer<< <fullBlocksPerGrid_startendindices, blockSize >> >(gridCellCount, dev_gridCellEndIndices  , -1);
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices);

  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
	kernCopyPosVelToCohPosVel << <fullBlocksPerGrid, blockSize>> > (numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_cohpos, dev_cohvel1);

  // - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchCoherent<< <fullBlocksPerGrid, blockSize>> >(numObjects, gridSideCount, gridMinimum, 
		gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_cohpos, dev_cohvel1, dev_cohvel2);

  // - Update positions
	//kernUpdatePos << < fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	kernUpdatePosCoh << < fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_cohpos, dev_cohvel2, dev_pos, dev_particleArrayIndices);
	kernCopyCohVel2ToVel1 << <fullBlocksPerGrid, blockSize>> > (numObjects, dev_particleArrayIndices, dev_cohvel2, dev_vel1);

  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_cohpos);
  cudaFree(dev_cohvel1);
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
