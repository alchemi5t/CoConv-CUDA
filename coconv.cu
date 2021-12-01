#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include <sstream>
#include <math.h>
#include <time.h>
#include <ctime>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <unistd.h>
#include <string.h>

// Turn on/off debug mode
#define DEBUG
#define FUNCTEST
#define PERFTEST

#ifdef DEBUG
#define LOG(...) printf(__VA_ARGS__); fflush(stdout);
#else
#define LOG(...) ;
#endif

unsigned int H, W, C, K = 3, maxDilation = 4, C_out;

#define TILE_WIDTH 32
#define TILE_HEIGHT 32

//Namespace for std
using namespace std;

//structure declaration for storing rows and columns for a matrix
struct matrix {
	unsigned int rows;  //storing rows of a matrix
	unsigned int cols;  //storing columns of a matrix
};

//handlerror declaration : to display file and line numbers of erroneous lines
static void HandleError( cudaError_t err, const char *file, int line ) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << " in " << file << " at line " << line << endl;
	}
}

//handle error alias name declaration
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


//global kernal for matrix multiplication, takes in input matrices and sizes, and multiplies them
//matrix multiplication is being done tile by tile
__global__ void matrix_mult(float* array1, unsigned int rows1, unsigned int cols1, float* array2, unsigned int rows2, unsigned int cols2, float* array3)
{
//shared memory takes one tile at a time
	__shared__ float S1[TILE_WIDTH][TILE_HEIGHT];   //to store tiles for array 1
	__shared__ float S2[TILE_HEIGHT][TILE_WIDTH];   //to store tiles for array 2

//threads x and y index for the current block
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	unsigned int c = blockIdx.x * blockDim.x + threadIdx.x; //row value using x-index of current thread
	unsigned int r = blockIdx.y * blockDim.y + threadIdx.y; //column value using y-index of current thread

	unsigned int idx = c * rows1 + r;       //column major index, using row and column value

	float val = 0;      //register to store multiplication result initialized to zero

	for (int m = 0; m < 1 + ((rows2 - 1) / TILE_WIDTH); m++) //going over all tiles one by one, with each m
	{

		int var1 = m * TILE_WIDTH + tx ; //x thread value for current tile
		int var2 = m * TILE_WIDTH + ty ; //y thread value for current tile

//copying a tile from array1
		if (r < rows1 && var1 < rows2)      //if the value is associated to a valid matrix coordinate in array1 then store it to shared memory S1
			S1[ty][tx] = array1[r + var1 * rows1]; //storing a "valid" value from array to shared memory
		else
			S1[ty][tx] = 0;                 //storing zero, since there is no valid value
		__syncthreads();                        //syncing all threads once shared memory S1 is stored

//copying a tile from array2
		if (c < cols2 && var2 < rows2)  //if value is associates to a valid matrix coordinate in array2 then store it to shared memory S2
			S2[ty][tx] = array2[var2 + rows2 * c]; //storing the valid value
		else
			S2[ty][tx] = 0;     //storing zero, since no valid value
		__syncthreads();        //synchronizing threads


		for (int i = 0; i < TILE_WIDTH; i++) //going over entire tile, ty row in S1 and tx column in S2
			val += S1[ty][i] * S2[i][tx]; //and multiplying elements
		__syncthreads();        //synchronizing threads

	}

	if (r < rows1 && c < cols2) //removing degenerate cases
		array3[idx] = val;  //saving multiplication result to global memory

}

float* gemm(float *array_A, float *array_B_gpu, int M_Arows, int M_Acols, int M_Brows, int M_Bcols) {

	float* array_C = (float*)malloc(M_Arows * M_Bcols * sizeof(float)); //array to store gpu result in column major format

	//float* array_D = (float*)malloc(M_Arows * M_Bcols * sizeof(float)); //arary to store cublas result in column major format

//GPU DEVICE PROPERTIES and selecting a GPU for calculation
	int nDevices;
	HANDLE_ERROR(cudaGetDeviceCount(&nDevices));

	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));    //using GPU0

//BLOCK AND GRID SIZE DECLARATION
	float thread_block = sqrt(prop.maxThreadsPerBlock); //2D blocks used
	dim3 DimGrid(ceil(M_Bcols / thread_block), ceil(M_Arows / thread_block), 1); //image saved as a 2D grid
	dim3 DimBlock(thread_block, thread_block, 1);

	size_t Sbytes = 2 * DimBlock.x * DimBlock.y ;   //2 arrays used in the calculation, hence 2 * DimBlock.x * DimBlock.y

//Checking if sufficient shared memory available or not

	if (prop.sharedMemPerBlock < Sbytes) {
		std::cout << "ERROR: insufficient shared memory" << std::endl;
		exit(1);
	}

//GPU MEMORY ALLOCATION
	//float *array_B_gpu;
	float *array_A_gpu, *array_C_gpu, *array_D_gpu;   //gpu arrays declared

	HANDLE_ERROR(cudaMalloc(&array_A_gpu, M_Arows * M_Acols * sizeof(float))); //allocate space to store arrayA

	//HANDLE_ERROR(cudaMalloc(&array_B_gpu, M_Brows * M_Bcols * sizeof(float))); //allocate space to store arrayB

	HANDLE_ERROR(cudaMalloc(&array_C_gpu, M_Arows * M_Bcols * sizeof(float))); //allocate space to store gpu result

	//HANDLE_ERROR(cudaMalloc(&array_D_gpu, M_Arows * M_Bcols * sizeof(float))); //allocate space to store cublas result


//COPY TO GPU MEMORY
	HANDLE_ERROR(cudaMemcpy(array_A_gpu, array_A, M_Arows * M_Acols * sizeof(float), cudaMemcpyHostToDevice)); //copy arrayA to gpu

	//HANDLE_ERROR(cudaMemcpy(array_B_gpu, array_B, M_Brows * M_Bcols * sizeof(float), cudaMemcpyHostToDevice)); //copy arrayB to gpu

	HANDLE_ERROR(cudaMemcpy(array_C_gpu, array_C, M_Arows * M_Bcols * sizeof(float), cudaMemcpyHostToDevice)); //copy arrayC to gpu

	//HANDLE_ERROR(cudaMemcpy(array_D_gpu, array_D, M_Arows * M_Bcols * sizeof(float), cudaMemcpyHostToDevice)); //copy arrayD to gpu


//time measurement for matrix multiplication
	cudaEvent_t start1, stop1;

	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

//MATRIX MULTIPLICATION USING KERNEL
	cudaEventRecord(start1);
	matrix_mult <<< DimGrid, DimBlock, Sbytes>>>(array_A_gpu, M_Arows, M_Acols, array_B_gpu, M_Brows, M_Bcols, array_C_gpu); //calling the kernel
	cudaEventRecord(stop1);

	cudaEventSynchronize(stop1);

	float milliseconds1 = 0, milliseconds2 = 0;//storing the execution time in milliseconds

	cudaEventElapsedTime(&milliseconds1, start1, stop1);//get the time in milliseconds
	LOG("  [!] Time taken by GPU GEMM %f ms\n", milliseconds1); //printing time taken by GPU

//copy to CPU MEMORY
	HANDLE_ERROR(cudaMemcpy(array_C, array_C_gpu, M_Arows * M_Bcols * sizeof(float), cudaMemcpyDeviceToHost)); //copying result of multiplication from gpu to cpu

/*	 float milliseconds1 = 0, milliseconds2 = 0;

//Creating handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

//parameter declaration for cublas implementation
	float alpha = 1.0;
	float beta = 0.0;

//cublas time measurement
	cudaEvent_t start2, stop2;

	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);

//MATRIX MULTIPLICATION USING CUBLAS
	cudaEventRecord(start2);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M_Arows, M_Bcols, M_Acols, &alpha, array_A_gpu, M_Arows, array_B_gpu, M_Brows, &beta, array_D_gpu, M_Arows);
	cudaEventRecord(stop2);
	
	cudaEventSynchronize(stop2);

	cudaEventElapsedTime(&milliseconds2, start2, stop2);//get the time in milliseconds
	LOG("  [!] Time taken by GPU CUBLAS %f ms\n", milliseconds2); //printing time taken by CUBLAS

//copy to CPU MEMORY

	HANDLE_ERROR(cudaMemcpy(array_D, array_D_gpu, M_Arows * M_Bcols * sizeof(float), cudaMemcpyDeviceToHost));*/
//copy result of multiplication using CUBLAS from gpu to cpu
	return array_C;

}
float* matrix_mult(float* array1, unsigned int rows1, unsigned int cols1, float* array2, unsigned int rows2, unsigned int cols2)
{
	float* C = (float*)malloc(rows1 * cols2 * sizeof(float));

//initailize the array to zero
	for (int idx = 0; idx < rows1 * cols2; idx++)
	{
		C[idx] = 0;
		int c = (int)(idx / rows1);
		int r = idx % rows1;

		for (int k = 0; k < rows2; k++)
		{
			C[idx] += array1[rows1 * k + r] * array2[rows2 * c + k];
		}

	}

	return C;

}


float* flatten_kernel(float * weights, int k, int d, int c_rows) {
	int c_cols = (k + (k - 1) * (d - 1)) * (k + (k - 1) * (d - 1));
	float *canvas = (float*)calloc(C * c_rows * c_cols, sizeof(float));
	int itr = 0;
	int k_id = 0;
	for (int dilation = 1; dilation <= d; dilation++) {
		int cur_kernel_size = k + (k - 1) * (dilation - 1);
		for (int kernel_id = 0; kernel_id < c_rows / 4; kernel_id++) {
			itr =    k_id * c_cols * C + (d - dilation) * pow(c_cols, 0.5) + (d - dilation);
			for (int weight_id = 0; weight_id < k * k * C; weight_id++) {
				canvas[itr] = weights[k_id * k * k * C + weight_id];
				itr++;
				if ((weight_id + 1) % (k * k) == 0) {
					itr = kernel_id * c_cols * C  + (dilation - 1) * (c_rows / 4) * c_cols * C +  ((weight_id + 1) / (k * k)) * c_cols + (d - dilation) * pow(c_cols, 0.5) + (d - dilation);

					continue;
				}
				if (((k_id * k * k * C + weight_id) + 1) % (k) == 0) {
					itr += (dilation - 1) * (pow(c_cols, 0.5)) + (pow(c_cols, 0.5) - (cur_kernel_size  ));
					/*for (int last_col_pads = 0; last_col_pads < (dilation - 1) * (pow(c_cols, 0.5)) + (pow(c_cols, 0.5) - (cur_kernel_size  )); last_col_pads++ ) {

						itr++;

					}*/
				}
				else {
					itr+=dilation-1;
					/*for (int inner_cols = 0; inner_cols < (dilation - 1); inner_cols++ ) {
						itr++;
					}*/


				}


			}
			k_id++;
		}
	}

	float *canvas_col = (float*)calloc(C * c_rows * c_cols, sizeof(float));
	itr = 0;
	for (int i = 0; i < C * c_cols; i++) {
		for (int j = 0; j < c_rows; j++) {
			canvas_col[itr] = canvas[(j * c_cols * C) + i];
			itr++;
		}
	}

	return canvas_col;

}


__global__ void flattenOnDevice(float * weights, float * canvas, int dilation, int c_rows, int c_cols, int d, int k_id, int k, int C, int kernel_id, int cur_kernel_size, int pow_c){

        int weight_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(weight_id/(k*k*C)<1){
        int itr=  ((weight_id % k) * (dilation - 1)) + (weight_id%(k*k)/k)*pow_c*(dilation - 1) + (weight_id % k) + (weight_id%(k*k)/k)*k;
        itr +=  (weight_id / (k*k)) * (kernel_id * c_cols * C  + (dilation - 1) * (c_rows / 4) * c_cols * C +  ((weight_id + 1) / (k * k)) * c_cols + (d - dilation) * pow_c + (d - dilation));
        itr += k_id * c_cols * C + (d - dilation) * pow_c + (d - dilation); //
//        itr = 0;
        canvas[itr] = weights[k_id * k * k * C + weight_id];
        }

}

float* flatten_kernelOnDevice(float * weights, int k, int d, int c_rows) {
	
	int c_cols = (k + (k - 1) * (d - 1)) * (k + (k - 1) * (d - 1));
	float *canvas = (float*)calloc(C * c_rows * c_cols, sizeof(float));
	float *canvas_d, *weights_d;
	const size_t sizeW = k * k * C * c_rows * sizeof(float);
	const size_t sizeC = C * c_rows * c_cols * sizeof(float);
	cudaMalloc((void**)&weights_d, sizeW);
	cudaMalloc((void**)&canvas_d, sizeC);
	cudaMemcpy(weights_d, weights, sizeW, cudaMemcpyHostToDevice);
	float thread_block = 1.0;
	int KERNELS = k*k*C;
	unsigned int GRID_SIZE = (KERNELS + thread_block - 1) / thread_block;
	int itr = 0;
	int k_id = 0;
	for (int dilation = 1; dilation <= d; dilation++) {
		int cur_kernel_size = k + (k - 1) * (dilation - 1);
		for (int kernel_id = 0; kernel_id < c_rows / 4; kernel_id++) {
			itr = k_id * c_cols * C + (d - dilation) * pow(c_cols, 0.5) + (d - dilation);
			flattenOnDevice <<< GRID_SIZE, thread_block>>>(weights_d,canvas_d,dilation,c_rows, c_cols, d,  k_id, k, C, kernel_id, cur_kernel_size, pow(c_cols, 0.5));
			cudaDeviceSynchronize();
			// for (int weight_id = 0; weight_id < k * k * C; weight_id++) {
			// }
			k_id++;
		}
	}
	cudaMemcpy(canvas, canvas_d, sizeC, cudaMemcpyDeviceToHost);
	float *canvas_col = (float*)calloc(C * c_rows * c_cols, sizeof(float));
	itr = 0;
	for (int i = 0; i < C * c_cols; i++) {
		for (int j = 0; j < c_rows; j++) {
			canvas_col[itr] = canvas[(j * c_cols * C) + i];
			itr++;
		}
	}

	return canvas_col;

}

void im2colOnHost(unsigned int n, float *matAc, float *matA, int H_, int W_, int L, int M, int K, int C)
{
// Using grid-stride loop if too big problem size.
// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
	for (int idx = 0;
	        idx < n;
	        idx += 1)
	{
		int m = (idx / C ) / L;
		int l = (idx / C ) % L;
		int r = idx % C;

// For each spatial position in output...
		if (m < M) {
			int w = m ;
			if (l < L) {
				int h = l;
// For each kernel weight...
				for (int x = 0; x < K; x++) {

					h = l;
					for (int y = 0; y < K; y++) {
						if (r < C) {
							matAc[((idx / C)*K * K * C) + (r * K * K) + (x * K + y)] = matA[(r * (H_ * W_)) + w * W_ + h];
							h++;
						}
					} w++;
				}
			}
		}
	}
}

// DEVICE KERNEL
// Takes matrix A [float *matA] and transforms it
// into column representation [float *matAc] on GPU
__global__
void im2colOnDevice(unsigned int n, float *matAc, float *matA, int H_, int W_, int L, int M, int K, int C)
{
// Using grid-stride loop if too big problem size.
// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//for (int idx = blockIdx.x * blockDim.x + threadIdx.x;idx < n;idx += blockDim.x * gridDim.x) {
	int m = (idx / C ) / L;
	int l = (idx / C ) % L;
	int r = idx % C;

// For each spatial position in output...
	if (m < M) {
		int w = m ;
		if (l < L) {
			int h = l;
// For each kernel weight...
			for (int x = 0; x < K; x++) {

				h = l;
				for (int y = 0; y < K; y++) {
					if (r < C) {
						matAc[((idx / C)*K * K * C) + (r * K * K) + (x * K + y)] = matA[(r * (H_ * W_)) + w * W_ + h];
						h++;
					}
				} w++;
			}
		}
	}
	//}
}


float* paddingFile(unsigned int maxDilation, char * fileName) {
	int newH = H + (2 * maxDilation);
	int newW = W + (2 * maxDilation);

	ifstream infile;
	infile.open(fileName);
	float *paddedInput = (float *)calloc(newH * newW * C, sizeof(float));
	for (int c = 0; c < C; c++) {
		for (int x = 0; x < H; x++) {
			for (int y = 0; y < W; y++) {
				infile >>  paddedInput[(c * newH * newW) + (((x + maxDilation)*newW) + y + maxDilation)];
			}

		}
	}
	return paddedInput;

}



float round6(float var)
{
// we use array of chars to store number
// as a string.
	char str[40];

// Print in string the value of var
// with two decimal point
	sprintf(str, "%.6f", var);

// scan string value in var
	sscanf(str, "%f", &var);

	return var;
}

double getTime(struct timeval start, struct timeval end)
{
	
	return ((double)(end.tv_usec - start.tv_usec) / 1000.0) + ((double)(end.tv_sec - start.tv_sec) * 1000.0);
}




void program(unsigned int blockSize = 0, unsigned int gridSize = 0)
{
// CONSTS AND VARIABLES
// Input/kernel/output counts and sizes
	LOG("  H=%d\tW=%d\tC=%d\tC_out=%d\n", H, W, C, C_out);
	
	const unsigned int countF = K * K * C;
	const size_t sizeF = countF * sizeof(float);
	const int paddedH = H + (2 * maxDilation);
	const int paddedW = W + (2 * maxDilation);
	const unsigned int L = H;
	const unsigned int M = W;
	const unsigned int KERNELS = L * M * C;
	//dilated kernel size
	const int K_ = K + (K - 1) * (maxDilation - 1);
	const unsigned int countF_ = K_ * K_ * C;
	const unsigned int countLR = L * M;
	const unsigned int countAc = countF_ * countLR;
	const size_t sizeAc = countAc * sizeof(float);



	// PREPARE DATA
	char * fileName = (char*) malloc(50 * sizeof(char));
	sprintf(fileName, "in_%d_%d_%d_%d", H, W, C, C_out);
	float *matInput = paddingFile(maxDilation, fileName);
	ifstream infile;
	char * fileNameW = (char*) malloc(50 * sizeof(char));
	sprintf(fileNameW, "weights_%d_%d_%d_%d", H, W, C, C_out);
	infile.open(fileNameW);
	float *matFlatten = (float *)malloc(sizeF * C_out);
	for (int i = 0; i < countF * C_out; i++) {
		infile >> matFlatten[i];
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	float thread_block =2.0;
	unsigned int GRID_SIZE = (KERNELS + thread_block - 1) / thread_block;
	LOG("  [!]GRID SIZE=%d\tBLOCK SIZE=%f\n", GRID_SIZE, thread_block);
	//TIME 
	struct timeval start, end,im2cols,im2cole,flattens,flattene,cpu_start, cpu_end, im2cols_cpu, im2cole_cpu, gemms_cpu, memcopys;
	
	// Alloc memory and copy data to device	
	float *devA, *devAc, *retAc, *retCpu;
	const size_t sizeI = C * paddedW * paddedH * sizeof(float);
	

	gettimeofday(&start, NULL);
	cudaMalloc((void**)&devA, sizeI);
	cudaMalloc((void**)&devAc, sizeAc);
	retAc = (float *)malloc(sizeAc);
	cudaMemcpy(devA, matInput, sizeI, cudaMemcpyHostToDevice);

	// Run im2col computation on device and copy results
	gettimeofday(&im2cols, NULL);
	im2colOnDevice <<< GRID_SIZE, thread_block>>>(KERNELS, devAc, devA, paddedH, paddedW, L, M, K_ , C);
	cudaDeviceSynchronize();
	gettimeofday(&im2cole, NULL);
	//cudaMemcpy(retAc, devAc, sizeAc, cudaMemcpyDeviceToHost);
	LOG("  [!] FINISHED CALCULATING im2col ON DEVICE %lfms\n",getTime(im2cols,im2cole));

	
	//GEMM
	gettimeofday(&flattens, NULL);
	float* kernelMatrix = flatten_kernel(matFlatten, K, maxDilation, C_out);
	gettimeofday(&flattene, NULL);
 	LOG("  [!] FINISHED CALCULATING Flatten ON DEVICE %lfms\n",getTime(flattens,flattene));
	float *res_gemm = gemm(kernelMatrix, devAc, C_out, countF_, K_ * K_ * C, countLR);
	gettimeofday(&end, NULL);
	LOG("  [!] FINISHED CALCULATING CoConv ON DEVICE %.16fms\n", getTime(start,end));

	//CPU
	gettimeofday(&cpu_start, NULL);
	retCpu = (float *)malloc(sizeAc);
	gettimeofday(&im2cols_cpu, NULL);
	im2colOnHost(KERNELS, retCpu, matInput, paddedH, paddedW, L, M, K_ , C);
	gettimeofday(&im2cole_cpu, NULL);
	LOG("  [!] FINISHED CALCULATING im2col ON CPU %lfms\n",getTime(im2cols_cpu,im2cole_cpu));

	gettimeofday(&gemms_cpu, NULL);
	float *res_gemm_cpu = matrix_mult(kernelMatrix, C_out, countF_, retCpu, K_ * K_ * C, countLR);
	gettimeofday(&cpu_end, NULL);
	LOG("  [!] FINISHED CALCULATING GEMM ON CPU %.16fms\n",getTime(gemms_cpu,cpu_end));
	LOG("  [!] FINISHED CALCULATING CoConv ON CPU %.16fms\n", getTime(cpu_start, cpu_end));
	

	//Speedup
	float speedup = 0.0;
	float cpu_time = getTime(cpu_start, cpu_end);
	float gpu_time = getTime(start,end);
	speedup = cpu_time / gpu_time;
	LOG("  SPEEDUP : %f", speedup);

	FILE * sp;
	sp = fopen("speedup", "a");
	fprintf(sp, "H=%d\tW=%d\tC=%d\tC_out=%d\tGPU_TIME=%f\tCPU_TIME=%f\tSPEEDUP=%f\t", H, W, C, C_out, gpu_time, cpu_time, speedup);
	ifstream output;
	
	float mse = 0.0;
	float mse_c = 0.0;

	char * fileN = (char*) malloc(50 * sizeof(char));
	sprintf(fileN, "op_%d_%d_%d_%d", H, W, C, C_out);
	output.open(fileN);
	
	FILE * fp;
	fp = fopen("out.txt", "w");

	for (int c = 0; c < C_out; c++) {
		for (int i = 0; i < countLR; i++) {
			int idx = i * C_out  + (c);
			float o = 0.0;
			output >> o;
			mse += (round6(o) - res_gemm[idx]) * (round6(o) - res_gemm[idx]);
			mse_c += (res_gemm[idx] - res_gemm_cpu[idx]) * (res_gemm[idx] - res_gemm_cpu[idx]);
			fprintf(fp, "%f\n", res_gemm[idx]);
		}
	}
	fclose(fp);
	mse_c /= countLR * C_out;
	mse /= countLR * C_out;
	LOG("\n  MSE: %f\tMSE CPU: %f\n", mse, mse_c);
	fprintf(sp, "MSE: %f\tMSE CPU: %f\n", mse, mse_c);
	fclose(sp);
    
    // CLEAN UP
	cudaFree(devA);
	cudaFree(devAc);
	free(matInput);
	free(matFlatten);
	free(retAc);
	free(retCpu);
	free(res_gemm);
	free(res_gemm_cpu);
}

int main(int argc, char * argv[])
{

	H = (unsigned int) atoi(argv[1]);
	W = (unsigned int) atoi(argv[2]);
	C = (unsigned int) atoi(argv[3]);
	C_out = (unsigned int) atoi(argv[4]);
	struct timeval t1, t2;
	double elapsedTime, totalTime = 0;
	gettimeofday(&t1, NULL);
	program();
	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
	elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
	totalTime += elapsedTime;				
	LOG("  [!] Whole program took %.3fms\n", totalTime);
	return EXIT_SUCCESS;
}

