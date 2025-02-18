#include <opencv2/opencv.hpp>
#include </usr/local/cuda/include/cuda_runtime.h>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;


__global__ void binarization_kernel(uchar*, uchar*, int, int, int);
__global__ void xor_kernel(uchar*, uchar*, uchar*, int, int);
__global__ void mark_differences_kernel(uchar*, uchar*, uchar*, int, int);

void printGPUinfo() {
    cudaDeviceProp device;
    cudaGetDeviceProperties(&device, 0);
    cout << "\n------------ GPU INFO ------------" << endl;
    cout << "GPU: " << device.name << endl;
    cout << "Max threads per block: " << device.maxThreadsPerBlock << endl;
}

/*********** MAIN ***********/
int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <img1> <img2> <threads_x> <threads_y>" << endl;
        return -1;
    }

    Mat img1 = imread(argv[1], IMREAD_COLOR);
    Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        cerr << "Error: Empty images!" << endl;
        return -1;
    }

    int th = 150;
    Mat display = img1.clone();
    cvtColor(img1, img1, COLOR_BGR2GRAY);

    dim3 NumThreadsPerBlock(atoi(argv[3]), atoi(argv[4])), nBlocks;

    nBlocks.x = img1.cols / NumThreadsPerBlock.x + ((img1.cols % NumThreadsPerBlock.x) == 0 ? 0 : 1);
    nBlocks.y = img1.rows / NumThreadsPerBlock.y + ((img1.rows % NumThreadsPerBlock.y) == 0 ? 0 : 1);
    
    printGPUinfo();
    
    
    size_t total_size = img1.rows * img1.cols * sizeof(uchar);
    size_t total_size_color = img1.rows * img1.cols * 3 * sizeof(uchar);
    uchar *d_img1, *d_img2, *d_bin1, *d_bin2, *d_xor, *d_display, *d_evidence;
    cudaMalloc(&d_img1, total_size);
    cudaMalloc(&d_img2, total_size);
    cudaMalloc(&d_bin1, total_size);
    cudaMalloc(&d_bin2, total_size);
    cudaMalloc(&d_xor, total_size);
    cudaMalloc(&d_display, total_size_color);
    cudaMalloc(&d_evidence, total_size_color);
    
    cudaMemcpy(d_img1, img1.data, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, img2.data, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_display, display.data, total_size_color, cudaMemcpyHostToDevice);

    auto start_time = steady_clock::now();
    
    binarization_kernel<<<nBlocks, NumThreadsPerBlock>>>(d_img1, d_bin1, img1.rows, img1.cols, th);
    binarization_kernel<<<nBlocks, NumThreadsPerBlock>>>(d_img2, d_bin2, img2.rows, img2.cols, th);
    xor_kernel<<<nBlocks, NumThreadsPerBlock>>>(d_bin1, d_bin2, d_xor, img1.rows, img1.cols);
    mark_differences_kernel<<<nBlocks, NumThreadsPerBlock>>>(d_xor, d_display, d_evidence, img1.rows, img1.cols);

    auto end_time = steady_clock::now();
    duration<double> elapsed_seconds = end_time - start_time;
    cout << "EXECUTION TIME: " << fixed << setprecision(6) << elapsed_seconds.count() << endl << endl;

    Mat xor_result(img1.size(), CV_8UC1);
    Mat evidence(img1.size(), CV_8UC3);
    cudaMemcpy(xor_result.data, d_xor, total_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(evidence.data, d_evidence, total_size_color, cudaMemcpyDeviceToHost);
    
    if (imwrite("outputs/evidence_result_parallel.jpg", evidence) && imwrite("outputs/XOR_parallel.jpg", xor_result)) {
        cout << "Images saved!\n";
    } else {
        cerr << "Error saving images!\n";
    }

    cudaFree(d_img1); 
    cudaFree(d_img2); 
    cudaFree(d_bin1);
    cudaFree(d_bin2); 
    cudaFree(d_xor); 
    cudaFree(d_display); 
    cudaFree(d_evidence);
    return 0;
}


/********** CUDA KERNELS **********/

__global__ void binarization_kernel(uchar* img, uchar* bin_result, int rows, int cols, int th) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y < rows && x < cols) {
        bin_result[y * cols + x] = (img[y * cols + x] >= th) ? 255 : 0;
    }
}

__global__ void xor_kernel(uchar* img1, uchar* img2, uchar* xor_result, int rows, int cols) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y < rows && x < cols) {
        int idx = y * cols + x;
        xor_result[idx] = img1[idx] ^ img2[idx];
    }
}

__global__ void mark_differences_kernel(uchar* xor_result, uchar* color_src, uchar* evidence, int rows, int cols) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y < rows && x < cols) {
        int idx = y * cols + x;
        if (xor_result[idx] > 0) {
            evidence[3 * idx] = color_src[3 * idx];
            evidence[3 * idx + 1] = color_src[3 * idx + 1];
            evidence[3 * idx + 2] = color_src[3 * idx + 2];
        }
    }
}
