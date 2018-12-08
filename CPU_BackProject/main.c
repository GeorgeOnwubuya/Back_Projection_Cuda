#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "support.h"
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
 *		INPUT DATA
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

/*****************************************************************************************
 * 		BACK PROJECTION
 ****************************************************************************************/

image Back_Projection(sinogram sino_output, float dist){

	//A function that does back projection without filtering
	//sinogram is the inout file created from matlab
	//sino_projection is the number of projections in the sinogram data file
	//sino_theta is the number of angles in the sinogram data file
	//dist is the distance between sensors

	image image_BP;
	image_BP.data = sino_output.data;
	image_BP.rows = sino_output.num_angles;
	image_BP.cols = sino_output.num_detectors;

	int  n;
	n = 2 * floor(image_BP.cols / (2 * sqrt(2)));
	image x_matrix;
	x_matrix.rows = n;
	x_matrix.cols = n;
	x_matrix.data = (float*)malloc((x_matrix.rows * x_matrix.cols)* sizeof(float));
	memset(x_matrix.data, 0, (n * n ) * sizeof(float));


	float cos_theta;
	float sin_theta;
	float angle_rad;
	int x_index;
	unsigned int sino_index;

	for(int angle = 0; angle < image_BP.rows; ++angle){

		angle_rad = pi / 180 * angle;
		cos_theta = cos(angle_rad);
    sin_theta = sin(angle_rad);

		for(int row = 0; row < n; ++row){

			for(int col = 0; col < n; ++col){

				x_index = row * n + col;
				sino_index = (unsigned int)(round((cos_theta * (col - n/2) + sin_theta * (row - n/2)) /
							   dist + image_BP.cols/2)) + image_BP.cols * angle;

				x_matrix.data[x_index] += image_BP.data[sino_index];
  			}
 		}
	}
	return x_matrix;
}

//Function to print data from array_sinogram
//Cross check it against matlab data
void print_nth_degree(float *sinogram, int n, int row, int col){

	 float *data;
         data  = sinogram + (n * row);
         for(int i = 0; i < row; ++i){
 		printf("[%3d] : %.6f\n", i, data[i]);
	 }
         puts("");
         return;
}

typedef struct{
	float *data;
	unsigned int rows;
	unsigned int cols;
}matrix;

//function to convert to matrix for sinogram
matrix output_conversion_sino(sinogram output_data){

	matrix convert;
	convert.data = output_data.data;
	convert.rows = output_data.num_detectors;
	convert.cols = output_data.num_angles;
	return convert;
}

//function to convert to matrix for image
matrix output_conversion_image(image output_data){

	matrix convert;
	convert.data = output_data.data;
	convert.rows = output_data.rows;
	convert.cols = output_data.cols;
	return convert;
}

//Function to print sinogram output
int write_data(const char *filename, matrix sino_out){

	if(filename == NULL){
		printf("file does not exist");
	return 1;
  	}
        int num_elms = sino_out.rows * sino_out.cols;
        FILE *file = fopen(filename, "wb");

	if(file == NULL){
		printf("file could not be opened");
	return 2;
	}

        size_t num_elements = fwrite((void*)sino_out.data, sizeof(float), num_elms, file);

        fclose(file);

	return 0;
}

int main() {

	/*********************************************************
		SINOGRAM DATA
	*********************************************************/
	//Print singram output data
	//View image on Matlab
	unsigned int detectors = 145;
	unsigned int angles    = 180;
	Timer timer;

	printf("Setting up sinogram data...");
    	fflush(stdout);
    	startTime(&timer);
	sinogram sinogram = read_sino_data("sinogram145x180_output.bin", angles, detectors);
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));
	//matrix conv_output = output_conversion_sino(output);
	//write_data("Sinogram_out_conv", conv_output);

	 /*********************************************************
                  BACK PROJECTION

	 *********************************************************/
	//The distance betweemn detectors is set to one
	printf("Launching back projection...");
    	fflush(stdout);
    	startTime(&timer);
	image image_recon = Back_Projection(sinogram, 1);
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));


	matrix conv_output = output_conversion_image(image_recon);

	printf("Writing image data...");
    	fflush(stdout);
    	startTime(&timer);
	write_data("Image_out", conv_output);
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));
	return 0;

}
