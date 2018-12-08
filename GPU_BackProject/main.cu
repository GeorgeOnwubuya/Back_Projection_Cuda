#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "support.h"
#include "kernel.cu"
#ifndef pi
#define pi 3.14159265358979323846
#endif 

//Struct to define sinogram output
typedef struct{
	float *data;
        unsigned int num_angles;
        unsigned int num_detectors;
}sinogram;
  
//Struct used to define the image output
typedef struct{
	float *data;
      	unsigned int rows;
      	unsigned int cols;
}image;

/***************************************************************
*              INPUT DATA
**************************************************************/
 
//Struct used to define input data.
//Sinogram data generated from matlab
//Sinogram is made up of number of detectors as its coloumns and 
//Converts binary sinogram data into array
sinogram read_sino_data(const char *filename, const unsigned int num_angles, const unsigned int num_detectors){
 
	//From sinogram struct
        sinogram sino;
        sino.num_detectors = num_detectors;
        sino.num_angles = num_angles;
 
        int num_elms = num_detectors * num_angles;
        FILE *file = fopen(filename, "r");
        sino.data = (float*)malloc(num_elms* sizeof(float));
        size_t num_elements = fread((void*)sino.data, sizeof(float), num_elms, file);
 
        fclose(file);
  
        return sino;
 }
 
int main(void){
	
	Timer timer;

	cudaFree(0);

	//Initialize host variables 
	const unsigned int NUM_DETECTORS = 1000;
	const unsigned int NUM_ANGLES    = 1035;
	const unsigned int BLOCK_SIZE    = 32;

	sinogram sinogram = read_sino_data("sinogram145x180_output.bin", NUM_ANGLES, NUM_DETECTORS);
	
	
	unsigned int n = 2 * floor(NUM_DETECTORS / (2 * sqrt(2)));
     
	cudaError_t cuda_ret;

	unsigned int num_elems = NUM_ANGLES * NUM_DETECTORS;

	//Allocate host variables
	float *image_out_h = (float*)malloc( sizeof(float) * num_elems);
 
        //Allocate device variables
        float *sinoData_d;
        float *image_out_d;
 
	printf("Allocating device variables..."); 
    	fflush(stdout);
    	startTime(&timer);

        cuda_ret = cudaMalloc((void**)&sinoData_d, sizeof(float) * num_elems);
	     	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**)&image_out_d, sizeof(float) * (n * n));
               	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory"); 
        cudaDeviceSynchronize();

	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));
         
	
	//Copying host variables to device
	printf("Copying device variables..."); 
    	fflush(stdout);
    	startTime(&timer);

        cuda_ret = cudaMemcpyAsync(sinoData_d, sinogram.data, sizeof(float) * num_elems, cudaMemcpyHostToDevice,0);
        	if(cuda_ret != cudaSuccess) FATAL("Unable to copy to device");
 	cudaDeviceSynchronize();
	
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));	

        //Set image device memory to zero
        cuda_ret = cudaMemset(image_out_d, 0, sizeof(float) * (n * n));
                if(cuda_ret != cudaSuccess) FATAL("Unable to set image memory");

        //Launching Kernel
        dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 grid((n + BLOCK_SIZE - 1)/BLOCK_SIZE, (n + BLOCK_SIZE - 1)/BLOCK_SIZE, 1);
	float dist = 1;
 	
	printf("Launching kernel..."); fflush(stdout);
    	startTime(&timer);	

       	backprojection<<< grid, block >>>(sinoData_d, dist, NUM_ANGLES, NUM_DETECTORS, image_out_d);
	//backprojection_reduceloopcalc<<< grid, block >>>(sinoData_d, dist, NUM_ANGLES, NUM_DETECTORS, image_out_d);
	//backprojection_reduce_numaccess<<< grid, block >>>(sinoData_d, dist, NUM_ANGLES, NUM_DETECTORS, image_out_d);
	
       	cuda_ret = cudaDeviceSynchronize();
        	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel, %d ", cuda_ret);
 	
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

        //Copy device variables from host
	printf("Copying data from device to host..."); fflush(stdout);
    	startTime(&timer);
	
        cuda_ret = cudaMemcpy(image_out_h, image_out_d, sizeof(float) * (n * n), cudaMemcpyDeviceToHost);
                if(cuda_ret != cudaSuccess) FATAL("Unable to copy to host");
	
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));	
 
        //Write data output to file
	printf("Writing data to file..."); fflush(stdout);
    	startTime(&timer);
	
        FILE *file = fopen("Image_gpu", "wb");
        fwrite((void*)image_out_h, sizeof(float), (n * n), file);
        fclose(file);

	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	//Free memory
        free(image_out_h);
        cudaFree(sinoData_d);
        cudaFree(image_out_d);
	
	return 0;
	
}












































