# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>ENTER YOUR NAME: NAVINKUMAR V</h3>
<h3>ENTER YOUR REGISTER NO: 212223230141</h3>
<h3>EX. NO: 03</h3>
<h3>DATE: 24/11/2026</h3>
<h1 align=center>  Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```
%%writefile sobelEdgeDetectionFilter.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage,unsigned char *dstImage,unsigned int width,unsigned int height)
{
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check image boundaries
    if (x >= 1 && x < width - 1 &&
        y >= 1 && y < height - 1)
    {
        // Sobel X and Y kernels
        int Gx[3][3] = {
            {-1,  0,  1},
            {-2,  0,  2},
            {-1,  0,  1}
        };

        int Gy[3][3] = {
            { 1,  2,  1},
            { 0,  0,  0},
            {-1, -2, -1}
        };

        int sumX = 0;
        int sumY = 0;

        // Apply Sobel filters
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                unsigned char pixel =
                    srcImage[(y + i) * width + (x + j)];

                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        // Calculate gradient magnitude
        int magnitude = sqrtf(sumX * sumX + sumY * sumY);

        // Clamp value between 0 and 255
        magnitude = min(max(magnitude, 0), 255);

        // Store the output pixel
        dstImage[y * width + x] =
            static_cast<unsigned char>(magnitude);
    }
}


void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Read input image
    Mat image = imread("/content/images.jpeg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate host memory for output image
    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);
    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage, image.data, imageSize, cudaMemcpyHostToDevice));

    // Define CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0));

    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockSize>>>(
    d_inputImage,
    d_outputImage,
    width,
    height
);
    cudaEventRecord(stop);

    // Synchronize events
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    // Write output image
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel.jpeg", outputImage);

    // Free memory
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print elapsed time
    printf("Total time taken: %f milliseconds\n", milliseconds);

    return 0;
}

```
## OUTPUT:
<img width="6000" height="3376" alt="IMAGES" src="https://github.com/user-attachments/assets/47d1cb03-b218-49ab-a4b0-703131b194f2" />


<img width="1672" height="941" alt="Sobel Edge Map of Spirit of Ecstasy" src="https://github.com/user-attachments/assets/649f1df7-2cd7-457d-881f-3a3080bc52b5" />


## RESULT:
Thus the program has been executed by using CUDA to<u> **0.154208  milliseconds.</u>**
<hr>
Questions:

1. What challenges did you face while implementing the Sobel filter for color images?<br>
```
Inserting images into Drive and find lines of bugs in CUDA program.
```
<hr>

2. How did changing the block size influence the performance of your CUDA implementation?<br>
Answer:
```Block size determines how many threads execute within each CUDA block.```
<hr>
<br><br>

3.What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.
<br> 
Answer: ```Minor differences may occur due to floating-point calculations and rounding.
Boundary pixels may differ depending on how they are initialized or processed.```
<br><hr>

4.Suggest potential optimizations for improving the performance of the Sobel filter.
<br> Answer: 
```
1. Reduce Memory Transfers: Minimize unnecessary data transfers between CPU and GPU.
2. Avoid Repeated Calculations: Optimize mathematical operations inside the CUDA kernel.
```
Deliverables:

Modified CUDA code with comments explaining your changes.
A report summarizing your findings, including graphs of execution times and a comparison of outputs.
Answers to the questions posed in the experiment.
Tools Required: **Google collab**

