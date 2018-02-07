#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>

/*
Somethings so confuse me, why i can't get same correct result every time.
*/

/*
These two kernel could be used on large array, but slow
Best advice: use __syncthreads() before you want to use different index
*/
__global__ void hillis_steele_scan_forward(float * d_out, float * d_in, const int array_size){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	d_out[idx] = d_in[idx];	
	for(int step = 1; step < array_size; step *= 2){		
		if(idx + step >= array_size) return;
		__syncthreads();
		float in1 = d_out[idx];
		__syncthreads();
		d_out[idx + step] += in1;
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
These two kernel could be used on small array, but fast
*/
__global__ void shared_hillis_steele_scan_forward(float *d_out, float *d_in, const int array_size) {
	extern __shared__ float sdata[];
	int idx = threadIdx.x;
	if(idx < array_size) {
		sdata[idx] = d_in[idx];
	} else {
		return;
	}	// the code below performs iterative scan on XY
	for(int step = 1; step < array_size; step *= 2){		
		if(idx + step >= array_size) return;
		__syncthreads();
		float in1 = sdata[idx];
		__syncthreads();
		sdata[idx + step] += in1;
	}
	d_out[idx] = sdata[idx];
}

__global__ void shared_hillis_steele_scan_backward(float * d_out, float * d_in, const int array_size){
	extern __shared__ float sdata[];
	int idx = threadIdx.x;
	if(idx < array_size) {
		sdata[idx] = d_in[idx];
	} else {
		return;
	}
	sdata[idx] = d_in[idx];
	for(int step = 1; step <= idx; step *= 2){		
		if(idx - step < 0) return;
		__syncthreads();
		float in1 = sdata[idx - step];
		__syncthreads();
		sdata[idx] += in1;
	}
	d_out[idx] = sdata[idx];
}

int main(int argc, char ** argv) {
	const int ARRAY_SIZE = 1025;
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
	//hillis_steele_scan_forward<<<numBlock, maxThreadPerBlock>>>(d_out, d_in, ARRAY_SIZE);
	//hillis_steele_scan_backward<<<numBlock, maxThreadPerBlock>>>(d_out, d_in);
	//shared_hillis_steele_scan_forward<<<numBlock, maxThreadPerBlock, maxThreadPerBlock  * sizeof(float)>>>(d_out, d_in, ARRAY_SIZE);
	shared_hillis_steele_scan_backward<<<numBlock, maxThreadPerBlock, maxThreadPerBlock  * sizeof(float)>>>(d_out, d_in, ARRAY_SIZE);

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
