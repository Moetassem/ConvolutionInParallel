#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "gputimer.h"
#include "wm.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#define MAX_NUMBER_THREADS (1024)

using namespace std;

cudaError_t imageConvolutionWithCuda(int numOfThreads, int weightBoxDim, char* inputImageName, char* outputImageName);

__global__ void convolutionKernel(unsigned char* inArray, float* outArray, float* wMs, int outputSizePerChannel, int numOfThreads, int width, int boxDim)
{
	for (int i = 0; i < outputSizePerChannel / numOfThreads; i++) {
		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
		int k = j + (boxDim - 1) * (j / (width - (boxDim - 1)));
		if (boxDim == 3) {
			outArray[j] = (float)(inArray[k] * wMs[0] + inArray[k + 1] * wMs[1] + inArray[k + 2] * wMs[2]
				+ inArray[k + width] * wMs[3] + inArray[k + width + 1] * wMs[4] + inArray[k + width + 2] * wMs[5]
				+ inArray[k + (2 * width)] * wMs[6] + inArray[k + (2 * width) + 1] * wMs[7] + inArray[k + (2 * width) + 2] * wMs[8]);
		}
		else if (boxDim == 5) {
			outArray[j] = (float)(inArray[k] * wMs[0] + inArray[k + 1] * wMs[1] + inArray[k + 2] * wMs[2] + inArray[k + 3] * wMs[3] + inArray[k + 4] * wMs[4]
				+ inArray[k + width] * wMs[5] + inArray[k + width + 1] * wMs[6] + inArray[k + width + 2] * wMs[7] + inArray[k + width + 3] * wMs[8] + inArray[k + width + 4] * wMs[9]
				+ inArray[k + (2 * width)] * wMs[10] + inArray[k + (2 * width) + 1] * wMs[11] + inArray[k + (2 * width) + 2] * wMs[12] + inArray[k + (2 * width) + 3] * wMs[13] + inArray[k + (2 * width) + 4] * wMs[14]
				+ inArray[k + (3 * width)] * wMs[15] + inArray[k + (3 * width) + 1] * wMs[16] + inArray[k + (3 * width) + 2] * wMs[17] + inArray[k + (3 * width) + 3] * wMs[18] + inArray[k + (3 * width) + 4] * wMs[19]
				+ inArray[k + (4 * width)] * wMs[20] + inArray[k + (4 * width) + 1] * wMs[21] + inArray[k + (4 * width) + 2] * wMs[22] + inArray[k + (4 * width) + 3] * wMs[23] + inArray[k + (4 * width) + 4] * wMs[24]);
		}
		else if (boxDim == 7) {
			outArray[j] = (float)(inArray[k] * wMs[0] + inArray[k + 1] * wMs[1] + inArray[k + 2] * wMs[2] + inArray[k + 3] * wMs[3] + inArray[k + 4] * wMs[4] + inArray[k + 5] * wMs[5] + inArray[k + 6] * wMs[6]
				+ inArray[k + width] * wMs[7] + inArray[k + width + 1] * wMs[8] + inArray[k + width + 2] * wMs[9] + inArray[k + width + 3] * wMs[10] + inArray[k + width + 4] * wMs[11] + inArray[k + width + 5] * wMs[12] + inArray[k + width + 6] * wMs[13]
				+ inArray[k + (2 * width)] * wMs[14] + inArray[k + (2 * width) + 1] * wMs[15] + inArray[k + (2 * width) + 2] * wMs[16] + inArray[k + (2 * width) + 3] * wMs[17] + inArray[k + (2 * width) + 4] * wMs[18] + inArray[k + (2 * width) + 5] * wMs[19] + inArray[k + (2 * width) + 6] * wMs[20]
				+ inArray[k + (3 * width)] * wMs[21] + inArray[k + (3 * width) + 1] * wMs[22] + inArray[k + (3 * width) + 2] * wMs[23] + inArray[k + (3 * width) + 3] * wMs[24] + inArray[k + (3 * width) + 4] * wMs[25] + inArray[k + (3 * width) + 5] * wMs[26] + inArray[k + (3 * width) + 6] * wMs[27]
				+ inArray[k + (4 * width)] * wMs[28] + inArray[k + (4 * width) + 1] * wMs[29] + inArray[k + (4 * width) + 2] * wMs[30] + inArray[k + (4 * width) + 3] * wMs[31] + inArray[k + (4 * width) + 4] * wMs[32] + inArray[k + (4 * width) + 5] * wMs[33] + inArray[k + (4 * width) + 6] * wMs[34]
				+ inArray[k + (5 * width)] * wMs[35] + inArray[k + (5 * width) + 1] * wMs[36] + inArray[k + (5 * width) + 2] * wMs[37] + inArray[k + (5 * width) + 3] * wMs[38] + inArray[k + (5 * width) + 4] * wMs[39] + inArray[k + (5 * width) + 5] * wMs[40] + inArray[k + (5 * width) + 6] * wMs[41]
				+ inArray[k + (6 * width)] * wMs[42] + inArray[k + (6 * width) + 1] * wMs[43] + inArray[k + (6 * width) + 2] * wMs[44] + inArray[k + (6 * width) + 3] * wMs[45] + inArray[k + (6 * width) + 4] * wMs[46] + inArray[k + (6 * width) + 5] * wMs[47] + inArray[k + (6 * width) + 6] * wMs[48]);
		}
		else {
			//outArray[j] = (float)inArray[j]; // not happening
			printf("you chose unsupported weight matrix dimension");
		}

		if (outArray[j] < 0) {
			outArray[j] = 0;
		}
		else if (outArray[j] > 255) {
			outArray[j] = 255;
		}
	}
}

//__global__ void pixelsCorrection(unsigned char* inArray, int sizeofInput, int numOfThreads)
//{
//	for (int i = 0; i < sizeofInput / numOfThreads; i++) {
//		int	k = (numOfThreads * i + threadIdx.x) + (blockIdx.x * numOfThreads);
//		if (inArray[k] < 0) {
//			inArray[k] = 0;
//		}
//		else if (inArray[k] > 255) {
//			inArray[k] = 255;
//		}
//		else {
//			inArray[k] = inArray[k];
//		}
//	}
//}

__global__ void pixelsSplitIntoQuarters(unsigned char* rgbaArray, unsigned char* rArray, unsigned char* gArray, unsigned char* bArray, unsigned char* aArray,
	int sizeofPixelsPerInputChannel, int numOfThreads)
{
	for (int i = 0; i < (sizeofPixelsPerInputChannel) / numOfThreads; i++) {
		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
		int k = j * 4;

		rArray[j] = rgbaArray[k];
		gArray[j] = rgbaArray[k + 1];
		bArray[j] = rgbaArray[k + 2];
		aArray[j] = rgbaArray[k + 3];
	}
}

__global__ void pixelsMerge(float* outrArray, float* outgArray, float* outbArray, float* outaArray, unsigned char* outfinalArray,
	int sizeofPixelsPerOutputChannel, int numOfThreads) {
	for (int i = 0; i < (sizeofPixelsPerOutputChannel / numOfThreads); i++) {
		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
		int k = 4 * j;

		outfinalArray[k] = outrArray[j];
		outfinalArray[k + 1] = outgArray[j];
		outfinalArray[k + 2] = outbArray[j];
		outfinalArray[k + 3] = 255;
	}
}

//__global__ void copyArrayKernel(unsigned char* inArray, float* outArray, int sizeofPixelsPerOutputChannel, int numOfThreads) {
//	for (int i = 0; i < (sizeofPixelsPerOutputChannel / numOfThreads); i++) {
//		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
//		outArray[j] = (float)inArray[j];
//		if (outArray[j] < 0) {
//			outArray[j] = 0;
//		}
//		else if (outArray[j] > 255) {
//			outArray[j] = 255;
//		}
//	}
//}

int main(int argc, char* argv[])
{
	char* inputImgName = "test.png";
	char* outImgName = "test_convolve_3.png";
	int weightMatDim = 3;
	int numOfThreads = 2049;

	/*if (argc != 6 || argv[1] == NULL || argv[2] == NULL || argv[3] == NULL || argv[4] == NULL ||
		argv[1] == "-h" || argv[1] == "--help" || argv[1] == "--h") {
		cout << "Assignment1.exe <Command> <name of input png> <name of output png> < # threads>" << endl;
		return 0;
	}
	else {
		if (argv[2] != NULL) {
			inputImgName = argv[2];
		}
		if (argv[3] != NULL) {
			outImgName = argv[3];
		}
		if (argv[4] != NULL) {
			numOfThreads = stoi(argv[4]);
		}
	}*/

	/*if (argv[1] != NULL && !strcmp(argv[1], "rectify")) {
		cout << "Rectifing" << endl;
		cudaError_t status = imageRectificationWithCuda(numOfThreads, inputImgName, outImgName);
	}

	if (argv[1] != NULL && !strcmp(argv[1], "pool")) {
		cout << "Pooling" << endl;
		cudaError_t status = imagePoolingWithCuda(numOfThreads, inputImgName, outImgName);
	}*/

	imageConvolutionWithCuda(numOfThreads, weightMatDim, inputImgName, outImgName);

	std::cout << "Name of Input Image File: " << inputImgName << std::endl;
	std::cout << "Name of Output Image File: " << outImgName << std::endl;
	std::cout << "Number of Threads: " << numOfThreads << std::endl;

	return 0;
}

cudaError_t imageConvolutionWithCuda(int numOfThreads, int weightBoxDim, char* inputImageName, char* outputImageName) {
	cudaError_t cudaStatus = cudaError_t::cudaErrorDeviceUninitilialized;
	//GpuTimer gpuTimer; // Struct for timing the GPU
	unsigned char* inputImage = nullptr;
	unsigned int width, height = 0;

	int error = lodepng_decode32_file(&inputImage, &width, &height, inputImageName);
	if (error != 0) {
		cout << "Failed to decode the image" << endl;
		cudaStatus = cudaError_t::cudaErrorAssert;
		goto Error;
	}

	int sizeOfArray = width * height * 4;
	int sizeOfOutputArray = (width - (weightBoxDim - 1)) * (height - (weightBoxDim - 1)) * 4;

	unsigned char* dev_RGBAArray, * dev_RArray, * dev_GArray, * dev_BArray, * dev_AArray, * dev_outArray;
	float* dev_outRArray, * dev_outGArray, * dev_outBArray, * dev_outAArray, * dev_wMs;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_RGBAArray, sizeOfArray * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	for (int i = 0; i < sizeOfArray; i++) {
		dev_RGBAArray[i] = inputImage[i];
	}

	// To make our life easier, we're going to split the RGBA values into separate arrays - let's start by mallocing them
	cudaStatus = cudaMallocManaged((void**)& dev_RArray, (sizeOfArray / 4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_GArray, (sizeOfArray / 4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_BArray, (sizeOfArray / 4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_AArray, (sizeOfArray / 4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outRArray, (sizeOfOutputArray / 4) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outGArray, (sizeOfOutputArray / 4) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outBArray, (sizeOfOutputArray / 4) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outAArray, (sizeOfOutputArray / 4) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outArray, (sizeOfOutputArray) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_wMs, weightBoxDim * weightBoxDim * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	for (int i = 0; i < (weightBoxDim); i++) {
		for (int j = 0; j < (weightBoxDim); j++) {
			if (weightBoxDim == 3) {
				dev_wMs[(i * weightBoxDim) + j] = w3[i][j];
			}
			else if (weightBoxDim == 5) {
				dev_wMs[(i * weightBoxDim) + j] = w5[i][j];
			}
			else if (weightBoxDim == 7) {
				dev_wMs[(i * weightBoxDim) + j] = w7[i][j];
			}
		}
	}

	int numBlocks = ((numOfThreads + (MAX_NUMBER_THREADS - 1)) / MAX_NUMBER_THREADS);
	int threadsPerBlock = ((numOfThreads + (numBlocks - 1)) / numBlocks);
	/*************************************** Parrallel Part of Execution **********************************************/
	//gpuTimer.Start();
	pixelsSplitIntoQuarters << <numBlocks, threadsPerBlock >> > (dev_RGBAArray, dev_RArray, dev_GArray, dev_BArray, dev_AArray, sizeOfArray / 4, threadsPerBlock);

	//Convolution of each array r,g,b,a
	convolutionKernel << <numBlocks, threadsPerBlock >> > (dev_RArray, dev_outRArray, dev_wMs, sizeOfOutputArray / 4, threadsPerBlock, width, weightBoxDim);

	convolutionKernel << <numBlocks, threadsPerBlock >> > (dev_GArray, dev_outGArray, dev_wMs, sizeOfOutputArray / 4, threadsPerBlock, width, weightBoxDim);

	convolutionKernel << <numBlocks, threadsPerBlock >> > (dev_BArray, dev_outBArray, dev_wMs, sizeOfOutputArray / 4, threadsPerBlock, width, weightBoxDim);

	//copyArrayKernel << <numBlocks, threadsPerBlock >> > (dev_AArray, dev_outAArray, sizeOfOutputArray / 4, threadsPerBlock);
	//convolutionKernel << <numBlocks, threadsPerBlock >> > (dev_AArray, dev_outAArray, dev_wMs, sizeOfOutputArray / 4, threadsPerBlock, width, weightBoxDim);

	//pixelsCorrection << <numBlocks, threadsPerBlock >> > (dev_outArray, sizeOfOutputArray, threadsPerBlock);

	pixelsMerge << <numBlocks, threadsPerBlock >> > (dev_outRArray, dev_outGArray, dev_outBArray, dev_outAArray, dev_outArray, sizeOfOutputArray / 4, threadsPerBlock);
	//gpuTimer.Stop();
	/*****************************************************************************************************************/
	//printf("-- Number of Threads: %d -- Execution Time (ms): %g \n", numOfThreads, gpuTimer.Elapsed());

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "convolutionKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching convolutionKernel!\n", cudaStatus);
		goto Error;
	}

	error = lodepng_encode32_file(outputImageName, dev_outArray, (width - (weightBoxDim - 1)), (height - (weightBoxDim - 1)));
	if (error != 0) {
		cout << "Failed to encode the image" << endl;
		cudaStatus = cudaError_t::cudaErrorAssert;
		goto Error;
	}

	free(inputImage);

Error:
	// BE FREE MY LOVLIES
	cudaFree(dev_RGBAArray);
	cudaFree(dev_RArray);
	cudaFree(dev_GArray);
	cudaFree(dev_BArray);
	cudaFree(dev_AArray);
	cudaFree(dev_outRArray);
	cudaFree(dev_outGArray);
	cudaFree(dev_outBArray);
	cudaFree(dev_outAArray);
	cudaFree(dev_outArray);

	return cudaStatus;
}