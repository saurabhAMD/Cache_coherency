#include <hip/hip_runtime.h>
#include "ex1.h"

// Element-Wise Sum of Two Arrays
__global__ void allreduceSum(int* sum, const int* dataA, const int* dataB, int size) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		sum[idx] = dataA[idx] + dataB[idx];
	}
	// if(idx==0)
	//printf("%d:%d, ",idx,sum[idx]);
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
	//printf("\n %d", allreduceArgs.currentGPU);

	// Allocate Memory for temporary buffer to store data at the current GPU
	// and copy data from the send buffer of current GPU to the temp buffer
	int* currentData;
	//hipMalloc(&currentData, allreduceArgs.N*sizeof(int));
	hipExtMallocWithFlags((void**)&currentData, allreduceArgs.N*sizeof(int), hipDeviceMallocFinegrained);
	// hipExtMallocWithFlags((void**)&currentData, allreduceArgs.N*sizeof(int), hipDeviceMallocUncached);
	hipMemcpy(currentData, allreduceArgs.sendBuffers[allreduceArgs.currentGPU], allreduceArgs.N * sizeof(int), hipMemcpyDeviceToDevice);

	// Perform an Allreduce Sum
	for(int i=0; i< allreduceArgs.numGPUs; i++) 
	{
		// Apply the sum on all the GPUs involved
		if (i != allreduceArgs.currentGPU)
		{
			//printf("\nGpu %d data being added\n", i);
			// Allocate memory for temporary buffer to store data to be added and 
			// copy data from the send buffer of GPU i to the temp buffer
			int* addData;
			//hipMalloc(&addData, allreduceArgs.N*sizeof(int));
			hipExtMallocWithFlags((void**)&addData, allreduceArgs.N*sizeof(int), hipDeviceMallocFinegrained);
			//hipExtMallocWithFlags((void**)&addData, allreduceArgs.N*sizeof(int), hipDeviceMallocUncached);
			hipMemcpy(addData, allreduceArgs.sendBuffers[i], allreduceArgs.N*sizeof(int), hipMemcpyDeviceToDevice);
			
			// Set Device to GPU being added
			hipSetDevice(i);

			// // Synchronize
			// hipDeviceSynchronize();
			
			// Call allreduceSum kernel and perform the allreduce sum
			dim3 blockDim(16);
			dim3 gridDim((allreduceArgs.N + blockDim.x - 1) / blockDim.x);
			allreduceSum<<<gridDim, blockDim>>>(currentData, currentData, addData, allreduceArgs.N);

			// // Synchronize
			// hipDeviceSynchronize();

			// Free temporary Buffer that stored the data being added
			hipFree(addData);
		}
	}

	// Copy to the receive buffer of the current GPU
	hipMemcpy(allreduceArgs.recvBuffers[allreduceArgs.currentGPU], currentData, allreduceArgs.N * sizeof(int), hipMemcpyDeviceToDevice);
	
	// Free temporary Buffer that stored the current GPU's Data
	hipFree(currentData);

}
