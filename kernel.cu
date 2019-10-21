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

using namespace std;

constexpr auto MAX_NUMBER_THREADS = 1024;

cudaError_t imageConvolutionWithCuda(int numOfThreads, int weightBoxDim, char* inputImageName, char* outputImageName);

__global__ void convolutionKernel(unsigned char* inArray, unsigned char* outArray, float* wM, int outputQuarterSize, int numOfThreads, int width, int boxDim)
{
	for (int i = 0; i < outputQuarterSize / numOfThreads; i++) {
		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
		int k = j + width * (j / width - (boxDim - 1));

		if (boxDim == 3) {
			outArray[j] = inArray[k] * wM[0] + inArray[k + 1] * wM[1] + inArray[k + 2] * wM[2]
						+ inArray[k + width] * wM[3] + inArray[k + width + 1] * wM[4] + inArray[k + width + 2] * wM[5]
						+ inArray[k + (2 * width)] * wM[6] + inArray[k + (2 * width) + 1] * wM[7] + inArray[k + (2 * width) + 2] * wM[8];
		}
		/*else if (boxDim == 5) {
			outArray[j] = inArray[k] * wM[0] + inArray[k + 1] * wM[1] + inArray[k + 2] * wM[2] + inArray[k + 3] * wM[3] + inArray[k + 4] * wM[4]
						+ inArray[k + width] * wM[5] + inArray[k + width + 1] * wM[6] + inArray[k + width + 2] * wM[7] + inArray[k + width + 3] * wM[8] + inArray[k + width + 4] * wM[9]
						+ inArray[k + (2 * width)] * wM[10] + inArray[k + (2 * width) + 1] * wM[11] + inArray[k + (2 * width) + 2] * wM[12] + inArray[k + (2 * width) + 3] * wM[13] + inArray[k + (2 * width) + 4] * wM[14]
						+ inArray[k + (3 * width)] * wM[15] + inArray[k + (3 * width) + 1] * wM[16] + inArray[k + (3 * width) + 2] * wM[17] + inArray[k + (3 * width) + 3] * wM[18] + inArray[k + (3 * width) + 4] * wM[19]
						+ inArray[k + (4 * width)] * wM[20] + inArray[k + (4 * width) + 1] * wM[21] + inArray[k + (4 * width) + 2] * wM[22] + inArray[k + (4 * width) + 3] * wM[23] + inArray[k + (4 * width) + 4] * wM[24];
		}*/
		/*else if (boxDim == 7) {
			outArray[j] = inArray[k] * wM[0] + inArray[k + 1] * wM[1] + inArray[k + 2] * wM[2] + inArray[k + 3] * wM[3] + inArray[k + 4] * wM[4]
				+ inArray[k + width] * wM[5] + inArray[k + width + 1] * wM[6] + inArray[k + width + 2] * wM[7] + inArray[k + width + 3] * wM[8] + inArray[k + width + 4] * wM[9]
				+ inArray[k + (2 * width)] * wM[10] + inArray[k + (2 * width) + 1] * wM[11] + inArray[k + (2 * width) + 2] * wM[12] + inArray[k + (2 * width) + 3] * wM[13] + inArray[k + (2 * width) + 4] * wM[14]
				+ inArray[k + (3 * width)] * wM[15] + inArray[k + (3 * width) + 1] * wM[16] + inArray[k + (3 * width) + 2] * wM[17] + inArray[k + (3 * width) + 3] * wM[18] + inArray[k + (3 * width) + 4] * wM[19]
				+ inArray[k + (4 * width)] * wM[20] + inArray[k + (4 * width) + 1] * wM[21] + inArray[k + (4 * width) + 2] * wM[22] + inArray[k + (4 * width) + 3] * wM[23] + inArray[k + (4 * width) + 4] * wM[24];
		}*/
	}
}

//__global__ void arrayMaxPerQuarterPixelKernel(unsigned char* inArray, unsigned char* outArray, int sizeofQuarterPixels, int numOfThreads, int width)
//{
//	for (int i = 0; i < ((sizeofQuarterPixels / 4) / numOfThreads); i++) {
//		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
//		int k = 2 * j + width * (j / (width / 2));
//
//		if (inArray[k] > inArray[k + 1]) {
//			outArray[j] = inArray[k];
//		}
//		else {
//			outArray[j] = inArray[k + 1];
//		}
//		if (inArray[k + width] > outArray[j]) {
//			outArray[j] = inArray[k + width];
//		}
//		if (inArray[k + width + 1] > outArray[j]) {
//			outArray[j] = inArray[k + width + 1];
//		}
//	}
//}

__global__ void pixelsSplitIntoQuarters(unsigned char* rgbaArray, unsigned char* rArray, unsigned char* gArray, unsigned char* bArray, unsigned char* aArray,
	int sizeofQuarterPixels, int numOfThreads)
{
	for (int i = 0; i < (sizeofQuarterPixels) / numOfThreads; i++) {
		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
		int k = j * 4;

		rArray[j] = rgbaArray[k];
		gArray[j] = rgbaArray[k + 1];
		bArray[j] = rgbaArray[k + 2];
		aArray[j] = rgbaArray[k + 3];
	}
}

__global__ void pixelsMerge(unsigned char* outrArray, unsigned char* outgArray, unsigned char* outbArray, unsigned char* outaArray, unsigned char* outfinalArray,
	int sizeofQuarterPixels, int numOfThreads) {
	for (int i = 0; i < ((sizeofQuarterPixels / 4) / numOfThreads); i++) {
		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
		int k = 4 * j;

		outfinalArray[k] = outrArray[j];
		outfinalArray[k + 1] = outgArray[j];
		outfinalArray[k + 2] = outbArray[j];
		outfinalArray[k + 3] = outaArray[j];
	}
}

int main(int argc, char* argv[])
{
	char* inputImgName = "testInput.png";
	char* outImgName = "testOutput.png";
	int weightMatDim = 3;
	int numOfThreads = 256;

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
	std::cout << "Name of Output Image File: " << numOfThreads << std::endl;

	return 0;
}

cudaError_t imageConvolutionWithCuda(int numOfThreads, int weightBoxDim, char* inputImageName, char* outputImageName) {
	cudaError_t cudaStatus = cudaError_t::cudaErrorDeviceUninitilialized;
	GpuTimer gpuTimer; // Struct for timing the GPU
	unsigned char* inputImage = nullptr;
	unsigned width, height = 0;

	int error = lodepng_decode32_file(&inputImage, &width, &height, inputImageName);
	if (error != 0) {
		cout << "You F**ed up decoding the image" << endl;
		cudaStatus = cudaError_t::cudaErrorAssert;
		goto Error;
	}

	int sizeOfArray = width * height * 4;
	int sizeOfOutputArray = (width - (weightBoxDim - 1)) * (height - (weightBoxDim - 1)) * 4;

	unsigned char* dev_RGBAArray, * dev_RArray, * dev_GArray, * dev_BArray, * dev_AArray, * dev_outRArray, * dev_outGArray, * dev_outBArray, * dev_outAArray, * dev_outArray;

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

	cudaStatus = cudaMallocManaged((void**)& dev_outRArray, (sizeOfOutputArray / 4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outGArray, (sizeOfOutputArray / 4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outBArray, (sizeOfOutputArray / 4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outAArray, (sizeOfOutputArray / 4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outArray, (sizeOfOutputArray) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	float w[9] = {1,2,-1,2,0.25,-2,1,-2,-1};
	int numBlocks = ((numOfThreads + (MAX_NUMBER_THREADS - 1)) / MAX_NUMBER_THREADS);
	int threadsPerBlock = ((numOfThreads + (numBlocks - 1)) / numBlocks);

	/*************************************** Parrallel Part of Execution **********************************************/
	gpuTimer.Start();
	pixelsSplitIntoQuarters << <numBlocks, threadsPerBlock >> > (dev_RGBAArray, dev_RArray, dev_GArray, dev_BArray, dev_AArray, sizeOfArray / 4, threadsPerBlock);

	//Convolution of each array r,g,b,a
	convolutionKernel << <numBlocks, threadsPerBlock >> > (dev_RArray, dev_outRArray, &w[0], sizeOfOutputArray / 4, threadsPerBlock, width, weightBoxDim);

	convolutionKernel << <numBlocks, threadsPerBlock >> > (dev_GArray, dev_outGArray, &w[0], sizeOfOutputArray / 4, threadsPerBlock, width, weightBoxDim);

	convolutionKernel << <numBlocks, threadsPerBlock >> > (dev_BArray, dev_outBArray, &w[0], sizeOfOutputArray / 4, threadsPerBlock, width, weightBoxDim);

	convolutionKernel << <numBlocks, threadsPerBlock >> > (dev_AArray, dev_outAArray, &w[0], sizeOfOutputArray / 4, threadsPerBlock, width, weightBoxDim);

	pixelsMerge << <numBlocks, threadsPerBlock >> > (dev_outRArray, dev_outGArray, dev_outBArray, dev_outAArray, dev_outArray, sizeOfArray / 4, threadsPerBlock);
	gpuTimer.Stop();
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

	error = lodepng_encode32_file(outputImageName, dev_outArray, width / 2, height / 2);
	if (error != 0) {
		cout << "You f**ed up encoding the image" << endl;
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