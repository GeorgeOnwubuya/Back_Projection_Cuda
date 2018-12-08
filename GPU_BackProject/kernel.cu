#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "support.h"
#ifndef pi
#define pi 3.14159265358979323846
#endif

/**************************************************************
                 KERNEL CODE
**************************************************************/
__global__ void backprojection(float *sinoData_d, float dist, unsigned int num_angles, unsigned int num_detectors, float *img_out_d ){
 
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
 
 	unsigned int  n;
	n = 2 * floorf(num_detectors / (2 * sqrtf(2)));
 
        float cos_theta;
 	float sin_theta;
 	float angle_rad;
 	int img_out_index;
 	unsigned sino_index;
  
	for(int angle = 0; angle < num_angles; ++angle){
 
	angle_rad = pi * angle / 180;
 	cos_theta = cosf(angle_rad);
 	sin_theta = sinf(angle_rad);
 
        	if((row < n) && (col < n)){

			img_out_index = row * n + col;
 			sino_index = static_cast<unsigned int>(rintf((cos_theta * (col - n/2.0f) + sin_theta * (row - n/2.0f)) / 
								dist + num_detectors/2.0f)) + num_detectors * angle;
 			img_out_d[img_out_index] += sinoData_d[sino_index];
 	        }
	}         
 }

//This kernel examines what happens when independent calcaulations are removed 
//from the loop. Calculations not dependent on loop(img_out_index = row * n + col;
__global__ void backprojection_reduceloopcalc(float *sinoData_d, float dist, unsigned int num_angles, unsigned int num_detectors, float *img_out_d){

	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int n;
	n = 2 * floor(num_detectors / (2 * sqrtf(2)));

	float cos_theta;
	float sin_theta;
	float angle_rad;
	int img_out_index;
	unsigned int sino_index;

	img_out_index = row * n + col;

	for(int angle = 0; angle < num_angles; ++angle){

		angle_rad = pi * angle / 180;
		cos_theta = cosf(angle_rad);
		sin_theta = sinf(angle_rad);

			if((row < n) && (col < n)){

				sino_index = static_cast<unsigned int>(rintf((cos_theta * (col - n/2.0f) + sin_theta * (row - n/2.0f)) /
                                                                  	dist + num_detectors/2.0f)) + num_detectors * angle;
                         	 img_out_d[img_out_index] += sinoData_d[sino_index];
			}
						
	}	
}

//This kernel reduces number of accesses to global memory
//Declare an array variable in local memory
__global__ void backprojection_reduce_numaccess(float *sinoData_d, float dist, unsigned int num_angles, unsigned int num_detectors, float *img_out_d){

	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int n;
	n = 2 * floor(num_detectors / (2 * sqrtf(2)));
  
        float cos_theta;
        float sin_theta;
        float angle_rad;
        int img_out_index;
        unsigned int sino_index;
  
	float pixel = 0;
	img_out_index = row * n + col;

	
	for(int angle = 0; angle < num_angles; ++angle){
 
        	angle_rad = pi * angle / 180;
        	cos_theta = cosf(angle_rad);
        	sin_theta = sinf(angle_rad);
  
                if((row < n) && (col < n)){
			 
                	sino_index = static_cast<unsigned int>(rintf((cos_theta * (col - n/2.0f) + sin_theta * (row - n/2.0f)) /
                                                                      dist + num_detectors/2.0f)) + num_detectors * angle;
                        pixel += sinoData_d[sino_index];
                }
  
	}
	img_out_d[img_out_index] = pixel;	 
}





	
	






