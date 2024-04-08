// Allreduce Exercise 2 - Ring Implementation

#include <iostream>
#include <vector>
#include <ctime>
#include <iomanip>

#include "ex1.h"
#include "ex1.cpp"

// Number of integers in the buffers
const int N = 10;
// const int N = 100;

int main() 
{
	// Initialize HIP
	hipInit(0);

	// Setup collectiveArgs with #GPUs, buffers, and size
	collectiveArgs allreduceArgs;
	
	// Set Size of Buffers
	allreduceArgs.N = N;
	
	int iterations;
	int total_iterations = 1;
	int data[N];
	
	// Get Number of GPUs
	hipGetDeviceCount(&allreduceArgs.numGPUs);
	if (allreduceArgs.numGPUs < 2) {
		std::cerr << "Error: At least two GPUs required for this test." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// Set send and recv Buffers for each GPU
	allreduceArgs.sendBuffers.resize(allreduceArgs.numGPUs);
	allreduceArgs.recvBuffers.resize(allreduceArgs.numGPUs);
	
	// Initialize data on each GPU that will be used in the reduction 
	for (int i = 0; i < allreduceArgs.numGPUs; ++i) 
	{
		hipSetDevice(i);

		hipMalloc(&allreduceArgs.sendBuffers[i], N * sizeof(int));
		hipMalloc(&allreduceArgs.recvBuffers[i], N * sizeof(int));

		// Initialize data to 0 for empty recvBuffers at this step
		for (int j = 0; j < N; j++)
		{
			data[j] = 0;
		}
		// Copy data to recvBuffer on GPU i from Host to Device to initialize recvBuffer
		hipMemcpy(allreduceArgs.recvBuffers[i], data, N * sizeof(int), hipMemcpyHostToDevice);
		
		// Initialize random data for GPU i
		for (int j = 0; j < N; j++)
		{
			data[j] = j;
		}
		// Copy data to sendBuffer on GPU i from Host to Device
		hipMemcpy(allreduceArgs.sendBuffers[i], data, N * sizeof(int), hipMemcpyHostToDevice);
	}

	// Begin Timer
	clock_t start_t = clock();

	// Perform an Allreduce on each GPU to store the SUM of
	// the data in the recvBuffers of each GPU
	for (iterations = 0; iterations < total_iterations; iterations++)
	{
		for (int i = 0; i < allreduceArgs.numGPUs; ++i) 
		{
			allreduceArgs.currentGPU = i;
			allreduce(allreduceArgs);
		}
	}

	//End Timer
	clock_t end_t = clock();

	//Calculate Total Time Spent in Allreduce
	double total_t = static_cast<double>(end_t - start_t) / CLOCKS_PER_SEC * 1e6;

	// Validate and Print the Results for Each GPU
	// --Print results before Allreduce Operation
	std::cout << "Data at each GPU BEFORE Allreduce Operation: ";
	for (int i = 0; i < N; ++i)
	{
		std::cout << data[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Data at each GPU AFTER Allreduce Operation: ";
	std::cout << std::endl;

	// --Print results after Allreduce Operation
	for (int i = 0; i < allreduceArgs.numGPUs; ++i) {
		hipSetDevice(i);

		int sum[N];
		hipMemcpy(sum, allreduceArgs.recvBuffers[i], N * sizeof(int), hipMemcpyDeviceToHost);

		std::cout << "GPU " << i << " has data: ";

		for (int j = 0; j < N; ++j) {
			std::cout << sum[j] << " ";
		}
		std::cout << std::endl;
	}


	std::cout << "Total time spent for Allreduce: " << std::fixed << std::setprecision(2) 
		<< total_t/total_iterations << " us." << std::endl;

	// Free and Cleanup
	for (int i = 0; i < allreduceArgs.numGPUs; ++i) {
		hipSetDevice(i);
		hipFree(allreduceArgs.sendBuffers[i]);
		hipFree(allreduceArgs.recvBuffers[i]);
	}

	return 0;
}
