#include <hip/hip_runtime.h>
#include "ex1.h"

// Element-Wise Sum of Two Arrays
__global__ void allreduceSum(int* sum, const int* data, int size)
{
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size) {
				// if(idx%2==0) sum[idx] = sum[idx];
                atomicAdd(&sum[idx],data[idx]);
				//sum[idx] = sum[idx] + data[idx];
        }
		printf("%d:%d, ",idx,sum[idx]);
}

void allreduce(collectiveArgs allreduceArgs) 
{
	/* TODO:
	 * Implement an allreduce operation utilizing the
	 * elements in the struct collectiveArgs found in 
	 * ex1.h. Use the sendbuffers of each GPU to pass
	 * to a GPU kernel where the sum of the sendbuffers
	 * at each GPU will then be copied to the recvbuffers
	 * of each GPU. Use allreduceSum to perform the addition
	 * of two arrays. 
	 */

	// Set Device to Current GPU
	hipSetDevice(allreduceArgs.currentGPU);
	printf("\n %d", allreduceArgs.currentGPU);

	// Copy data of current GPU send buffer to it's receive buffer
	hipMemcpy(allreduceArgs.recvBuffers[allreduceArgs.currentGPU], allreduceArgs.sendBuffers[allreduceArgs.currentGPU], allreduceArgs.N * sizeof(int), hipMemcpyDeviceToDevice);
	
	// Perform an Allreduce Sum
	for(int i=0; i< allreduceArgs.numGPUs; i++) 
	{
		// Apply the sum on all the GPUs involved
		if (i != allreduceArgs.currentGPU)
		{
			// Allocate memory for temporary buffer to store data to be added and 
			// copy data from the send buffer of GPU i to the temp buffer
			printf("\nGpu %d data being added\n", i);
			int* addData;
			hipMalloc(&addData, allreduceArgs.N*sizeof(int));
			//hipExtMallocWithFlags((void**)&addData, allreduceArgs.N*sizeof(int), hipDeviceMallocFinegrained);
			hipMemcpy(addData, allreduceArgs.sendBuffers[i], allreduceArgs.N*sizeof(int), hipMemcpyDeviceToDevice);
			
			
			// Set Device to GPU being added
			hipSetDevice(i);

			// // Synchronize
			// hipDeviceSynchronize();
			
			// Call allreduceSum kernel and perform the allreduce sum
			//dim3 blockDim(256);
			dim3 blockDim(16);
			dim3 gridDim((allreduceArgs.N + blockDim.x - 1) / blockDim.x);
			allreduceSum<<<gridDim, blockDim>>>(allreduceArgs.recvBuffers[allreduceArgs.currentGPU], addData, allreduceArgs.N);

			// // Synchronize
			//hipDeviceSynchronize();

			// Free temporary Buffer that stored the data being added
			hipFree(addData);
		}
	}
}
