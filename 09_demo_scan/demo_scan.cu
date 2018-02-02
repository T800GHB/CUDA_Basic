#include <iostream>
#include <numeric>
#include <stdlib.h>

#include <stdio.h>

/*
The logic in the two kernel is correct, but when you apply on a large array.
Apply on a small array will generate correct result.
*/
__global__ void hillis_steele_scan_forward(float * d_out, float * d_in, const int array_size){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	d_out[idx] = d_in[idx];
	for(int step = 1; step < array_size; step *= 2){		
		if(idx + step >= array_size) return;
		__syncthreads();
		d_out[idx + step] = d_out[idx] + d_out[idx + step];
		__syncthreads();
	}
}

__global__ void hillis_steele_scan_backward(float * d_out, float * d_in){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	d_out[idx] = d_in[idx];
	for(int step = 1; step <= idx; step *= 2){		
		if(idx - step < 0) return;
		__syncthreads();
		float in1 = d_out[idx - step];
		__syncthreads();
		d_out[idx] += in1;
	}
}

/*
I don't know how to understand the index in this kernle.
This kernel will generate correct result, when the input array is large
But this kernel will output incorrect result, when the input array is small.
*/
__global__ void work_inefficient_scan_kernel(float *X, float *Y, int InputSize) {
	extern __shared__ float XY[];
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < InputSize) {
	XY[threadIdx.x] = X[i];
	}
	// the code below performs iterative scan on XY
	for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
	__syncthreads();
	XY[threadIdx.x] += XY[threadIdx.x-stride];
	}
	Y[i] = XY[threadIdx.x];
}

int main(int argc, char ** argv) {
	const int ARRAY_SIZE = 230;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	const int maxThreadPerBlock = 512;
	const int numBlock = ARRAY_SIZE / maxThreadPerBlock + 1;
	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = float(i);
	}
	float h_out[ARRAY_SIZE];

	// declare GPU memory pointers
	float * d_in;
	float * d_out;

	// allocate GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	hillis_steele_scan_forward<<<numBlock, maxThreadPerBlock>>>(d_out, d_in, ARRAY_SIZE);
	//hillis_steele_scan_backward<<<numBlock, maxThreadPerBlock>>>(d_out, d_in);
	//work_inefficient_scan_kernel<<<numBlock, maxThreadPerBlock, maxThreadPerBlock * sizeof(float)>>>(d_out, d_in, ARRAY_SIZE);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
