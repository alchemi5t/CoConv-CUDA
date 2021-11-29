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
// Feature maps dimensionality descriptions and assumptions:
//             : Height          : Width           : Channels  : Number                    :
// INPUT   / A | H               | W               | C         | ------------------------- |  
// KERNELS / F | P = K           | Q = K           | R = C     | D = number of kernels = 1 |
// OUTPUT  / B | L = H * (K - 1) | M = W * (K - 1) | N = D = 1 | ------------------------- |
// [!] K must be odd number.
// [!] Data layout for INPUT/OUTPUT: C x H x W.
// [!] Data layout for KERNELS: D x R(=C) x P(=K) x Q(=K)

// Turn on/off debug mode
 #define DEBUG
 #define FUNCTEST
#define PERFTEST

#ifdef DEBUG
    #define LOG(...) printf(__VA_ARGS__); fflush(stdout);
#else
    #define LOG(...) ;
#endif

const unsigned int H = 4, W = 4, C = 1, K = 3; 


#define TILE_WIDTH 32
    #define TILE_HEIGHT 32
    
    //Namespace for std
    using namespace std;

    //structure declaration for storing rows and columns for a matrix
    struct matrix{
        unsigned int rows;  //storing rows of a matrix
        unsigned int cols;  //storing columns of a matrix
    };

    //handlerror declaration : to display file and line numbers of erroneous lines
    static void HandleError( cudaError_t err, const char *file, int line ) {
        if (err != cudaSuccess) {
            cout<<cudaGetErrorString(err)<<" in "<< file <<" at line "<< line<<endl;
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
        unsigned int tx=threadIdx.x;    
        unsigned int ty=threadIdx.y;

        unsigned int c=blockIdx.x*blockDim.x + threadIdx.x; //row value using x-index of current thread
        unsigned int r=blockIdx.y*blockDim.y + threadIdx.y; //column value using y-index of current thread

        unsigned int idx=c*rows1+r;             //column major index, using row and column value
        
        float val=0;        //register to store multiplication result initialized to zero

        for(int m=0; m<1+((rows2-1)/TILE_WIDTH);m++)    //going over all tiles one by one, with each m
        {

            int var1=m*TILE_WIDTH+tx ;      //x thread value for current tile
            int var2=m*TILE_WIDTH+ty ;      //y thread value for current tile
            
            //copying a tile from array1
            if (r < rows1 && var1 < rows2)      //if the value is associated to a valid matrix coordinate in array1 then store it to shared memory S1
                S1[ty][tx]=array1[r + var1*rows1];//storing a "valid" value from array to shared memory
            else
                    S1[ty][tx]=0;                   //storing zero, since there is no valid value
                __syncthreads();                        //syncing all threads once shared memory S1 is stored
            
            //copying a tile from array2
                if(c < cols2 && var2 < rows2)   //if value is associates to a valid matrix coordinate in array2 then store it to shared memory S2
                    S2[ty][tx]=array2[var2+rows2*c];    //storing the valid value
                else 
                    S2[ty][tx]=0;       //storing zero, since no valid value
            __syncthreads();        //synchronizing threads
            

            for(int i=0; i<TILE_WIDTH;i++)  //going over entire tile, ty row in S1 and tx column in S2
                val+=S1[ty][i]*S2[i][tx];   //and multiplying elements
            __syncthreads();        //synchronizing threads

        }
        
        if(r < rows1 && c< cols2)   //removing degenerate cases
            array3[idx]=val;    //saving multiplication result to global memory
            
    }

    void gemm(float *array_A, float *array_B, int M_Arows, int M_Acols, int M_Brows, int M_Bcols) {

        float* array_C=(float*)malloc(M_Arows*M_Bcols*sizeof(float));//array to store gpu result in column major format
        
        float* array_D=(float*)malloc(M_Arows*M_Bcols*sizeof(float));//arary to store cublas result in column major format

        //GPU DEVICE PROPERTIES and selecting a GPU for calculation
        int nDevices;
        HANDLE_ERROR(cudaGetDeviceCount(&nDevices));

        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));    //using GPU0

        //BLOCK AND GRID SIZE DECLARATION
        float thread_block=sqrt(prop.maxThreadsPerBlock);   //2D blocks used
        dim3 DimGrid(ceil(M_Bcols/thread_block),ceil(M_Arows/thread_block),1); //image saved as a 2D grid
        dim3 DimBlock(thread_block,thread_block,1);

        size_t Sbytes = 2* DimBlock.x * DimBlock.y ;    //2 arrays used in the calculation, hence 2 * DimBlock.x * DimBlock.y
        
        //Checking if sufficient shared memory available or not

        if(prop.sharedMemPerBlock < Sbytes){
            std::cout<<"ERROR: insufficient shared memory"<<std::endl;
            exit(1);
        }

        //GPU MEMORY ALLOCATION
        float *array_A_gpu, *array_B_gpu, *array_C_gpu, *array_D_gpu;   //gpu arrays declared
       
        HANDLE_ERROR(cudaMalloc(&array_A_gpu,M_Arows*M_Acols*sizeof(float))); //allocate space to store arrayA

        HANDLE_ERROR(cudaMalloc(&array_B_gpu,M_Brows*M_Bcols*sizeof(float))); //allocate space to store arrayB

        HANDLE_ERROR(cudaMalloc(&array_C_gpu,M_Arows*M_Bcols*sizeof(float))); //allocate space to store gpu result

        HANDLE_ERROR(cudaMalloc(&array_D_gpu,M_Arows*M_Bcols*sizeof(float))); //allocate space to store cublas result


        //COPY TO GPU MEMORY
        HANDLE_ERROR(cudaMemcpy(array_A_gpu, array_A, M_Arows*M_Acols*sizeof(float), cudaMemcpyHostToDevice));//copy arrayA to gpu

        HANDLE_ERROR(cudaMemcpy(array_B_gpu, array_B, M_Brows*M_Bcols*sizeof(float), cudaMemcpyHostToDevice));//copy arrayB to gpu

        HANDLE_ERROR(cudaMemcpy(array_C_gpu, array_C, M_Arows*M_Bcols*sizeof(float), cudaMemcpyHostToDevice));//copy arrayC to gpu

        HANDLE_ERROR(cudaMemcpy(array_D_gpu, array_D, M_Arows*M_Bcols*sizeof(float), cudaMemcpyHostToDevice));//copy arrayD to gpu


        //time measurement for matrix multiplication
        cudaEvent_t start1, stop1;
        
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);
        
        //MATRIX MULTIPLICATION USING KERNEL
        cudaEventRecord(start1);
        matrix_mult<<<DimGrid, DimBlock, Sbytes>>>(array_A_gpu,M_Arows,M_Acols,array_B_gpu,M_Brows,M_Bcols,array_C_gpu);//calling the kernel
        cudaEventRecord(stop1);

        cudaEventSynchronize(stop1);

        float milliseconds1 = 0, milliseconds2 = 0;//storing the execution time in milliseconds
        
        cudaEventElapsedTime(&milliseconds1, start1, stop1);//get the time in milliseconds
        cout<<"Time taken by GPU = "<<milliseconds1<<" ms"<<endl;//printing time taken by GPU

        //copy to CPU MEMORY
        HANDLE_ERROR(cudaMemcpy(array_C, array_C_gpu, M_Arows*M_Bcols*sizeof(float), cudaMemcpyDeviceToHost));//copying result of multiplication from gpu to cpu

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
        cout<<"Time taken by CUBLAS= "<<milliseconds2<<" ms"<<endl;//printing time taken by CUBLAS
        
        //copy to CPU MEMORY
        
            HANDLE_ERROR(cudaMemcpy(array_D, array_D_gpu, M_Arows*M_Bcols*sizeof(float), cudaMemcpyDeviceToHost));//copy result of multiplication using CUBLAS from gpu to cpu

        //CALCULATING MEAN SQUARED ERROR IN BOTH METHODS OF MATRIX MULTIPLICATION
        float mse=0; //mean squared error;

        for(int i=0; i<M_Arows*M_Bcols;i++) {
            mse=mse+(array_C[i]-array_D[i])*(array_C[i]-array_D[i]);//calculating element by element
            printf("%.3f ", array_C[i]);
        }
        mse=mse/(M_Arows*M_Bcols);  //taking the mean of squared error
            
        cout<<endl<<"Mean square error = "<<mse<<endl;//printing out the mean squared error


    }




// HOST FUNCTION
// Takes matrix A [float *matA] and transforms it
// into column representation [float *matAc]
void im2colOnHost(float *matA, float *matAc, int radiusF, int countF, int L, int M, int K, int C)
{
    // For each spatial position in output...
    for (int m = 0; m < M; m++) {
        int w = m + radiusF;
        for (int l = 0; l < L; l++) {
            int h = l + radiusF;

            // Progress..
            LOG("\r[i] Calculation on CPU %3d%%...", ((m * L + l) * 100 / (M * L)));

            // For each kernel weight...
            for (int q = 0, oq = -1 * radiusF; oq <= radiusF; q++, oq++) {
                for (int p = 0, op = -1 * radiusF; op <= radiusF; p++, op++) {
                    for (int r = 0; r < C; r++) {
                        matAc[(r + C * (p + K * q)) + countF * (l + L * m)] = matA[r + C * ((h + op) + H * (w + oq))]; 
                        LOG("matAc[%3d x %3d] <- matA[%3d x %3d x %3d]\n", (r + C * (p + K* q)), (l + L * m), (h + op), (w + oq), r);
                    }
                }
            }
        }
    }
    LOG("\n");
}
 
// DEVICE KERNEL
// Takes matrix A [float *matA] and transforms it
// into column representation [float *matAc] on GPU
__global__ 
void im2colOnDevice(unsigned int n, float *matAc, float *matA, int radiusF, int countF, int L, int M, int K, int C)
{
    // Using grid-stride loop if too big problem size.
    // https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) 
    {
        int m = (idx / C) / L;
        int l = (idx / C) % L;
        int r = idx % C;
        
        // For each spatial position in output...
        if (m < M) {
            int w = m + radiusF;
            if (l < L) {
                int h = l + radiusF;
                // For each kernel weight...
                for (int q = 0, oq = -1 * radiusF; oq <= radiusF; q++, oq++) {
                    for (int p = 0, op = -1 * radiusF; op <= radiusF; p++, op++) {
                        if (r < C) {
                            matAc[(r + C * (p + K * q)) + countF * (l + L * m)] = matA[r + C * ((h + op) + H * (w + oq))]; 
                        }
                    }
                }
            }
        }
    }
}
 
// DEVICE KERNEL
// Takes matrix A [float *matA] and transforms it
// into column representation [float *matAc] on GPU
__global__ 
void col2imOnDevice(unsigned int n, float *matA, float *matAc, int radiusF, int countF, int L, int M, int K, int C)
{
    // Using grid-stride loop if too big problem size.
    // https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) 
    {
        int m = (idx / C) / L;
        int l = (idx / C) % L;
        int r = idx % C;
    
        // For each spatial position in output...
        if (m < M) {
            int w = m + radiusF;
            if (l < L) {
                int h = l + radiusF;
                // For each kernel weight...
                for (int q = 0, oq = -1 * radiusF; oq <= radiusF; q++, oq++) {
                    for (int p = 0, op = -1 * radiusF; op <= radiusF; p++, op++) {
                        if (r < C) {
                            matA[r + C * ((h + op) + H * (w + oq))] = matAc[(r + C * (p + K * q)) + countF * (l + L * m)]; 
                        }
                    }
                }
            }
        }
    }
}

void program(unsigned int blockSize, unsigned int gridSize = 0)
{
    // CONSTS AND VARIABLES

    // Input/kernel/output counts and sizes
    const unsigned int countA = H*W*C;
    const size_t sizeA = countA*sizeof(float);
    LOG("[i] INPUT PARAMS: %u height, %u width, %u channels, %u elems, %u bytes\n", H, W, C, countA, sizeA);

    const unsigned int radiusF = (K - 1) / 2;
    const unsigned int countF = K*K*C;
    const size_t sizeF = countF*sizeof(float);
    LOG("[i] FILTER PARAMS: %u radius, %u elems, %u bytes\n", radiusF, countF, countF*sizeof(float));
    
    const unsigned int L = H - (K - 1);
    const unsigned int M = W - (K - 1);
    LOG("[i] OUTPUT PARAMS: %u height, %u width, %u channels\n", L, M, 1);
    
    const unsigned int countLR = L * M;
    const unsigned int countAc = countF * countLR;
    const size_t sizeAc = countAc*sizeof(float);
    LOG("[i] INPUT IN COL PARAMS: %u elems, %u bytes\n", countAc, sizeAc);

    
    // PREPARE DATA

    // Generate input data
    float *matA = (float *)malloc(sizeA);
    for (int i = 0; i < countA; i++) {
        matA[i] =(float)(i+1);
	printf("%.1f ",matA[i]);
	if((i+1)%4==0){
	printf("\n");}
    }
    LOG("  [!] FINISHED GENERATING INPUT\n");

#ifdef FUNCTEST
    // Calculate im2col result
    float *matAc = (float *)malloc(sizeAc);
    im2colOnHost(matA, matAc, radiusF, countF, L, M, K, C);
    LOG("  [!] FINISHED CALCULATING im2col RESULT ON CPU\n");
for (int i = 0; i < countAc; i++) {
        printf("%.1f ",matAc[i]);
        if((i+1)%9==0)
{printf("\n");}
    }
#endif


    // Alloc memory and copy data to device
    float *devA, *devAc, *retAc;
    
    cudaMalloc((void**)&devA, sizeA); 
    cudaMalloc((void**)&devAc, sizeAc); 
    retAc = (float *)malloc(sizeAc);

    cudaMemcpy(devA, matA, sizeA, cudaMemcpyHostToDevice); 

    // Compute default grid size if it wasn't passed
    const unsigned int KERNELS_NUM = L * M * C;
    if (gridSize == 0)
        gridSize = (KERNELS_NUM + blockSize - 1) / blockSize;
    
    // Run im2col computation on device and copy results
 	struct timeval t3, t4;
gettimeofday(&t3, NULL);
    im2colOnDevice<<<gridSize, blockSize>>>(KERNELS_NUM, devAc, devA, radiusF, countF, L, M, K, C);
gettimeofday(&t4, NULL);
    LOG("  [!] FINISHED CALCULATING im2col ON DEVICE %.16fms\n",(t4.tv_usec-t3.tv_usec)/1000.0+(t4.tv_sec-t3.tv_sec)*1000.0);
    
    cudaMemcpy(retAc, devAc, sizeAc, cudaMemcpyDeviceToHost);
	for (int i = 0; i < countAc; i++) {
        printf("%.1f ",retAc[i]);
	if((i+1)%9==0)
{printf("\n");}
    }
printf("\n");

//GEMM
float *matF = (float *)malloc(sizeF);
for (int i = 0; i < countF; i++) {
        matF[i] =(float)(i+1);
    }
gemm(retAc, matF, (L*M), (K*K*C), (K*K*C), 1);

#ifdef FUNCTEST
    // Compare results
    int success = 1;
    for (int i = 0; i < countAc; i++) {
        if (retAc[i] != matAc[i]) {
            success = 0;
            printf("TEST FAILED: im2col device kernel...\n");
            break;
        }
    }

    if (success) {
        printf("TEST PASSED: im2col device kernel!\n");
    }
#endif

    // Allocate memory for return value
    float *retA;
    retA = (float *)malloc(sizeA);
    cudaMemset(devA, 0, sizeA); 
    
    // Run col2im computation on device and copy results
gettimeofday(&t3, NULL);    
col2imOnDevice<<<gridSize, blockSize>>>(KERNELS_NUM, devA, devAc, radiusF, countF, L, M, K, C);
gettimeofday(&t4, NULL);    
LOG("  [!] FINISHED CALCULATING col2im ON DEVICE %.16fms\n",(t4.tv_usec-t3.tv_usec)/1000.0+(t4.tv_sec-t3.tv_sec)*1000.0);
    
    cudaMemcpy(retA, devA, sizeA, cudaMemcpyDeviceToHost);

#ifdef FUNCTEST
    // Compare results
    success = 1;
    for (int i = 0; i < countA; i++) {
        if (retA[i] != matA[i]) {
            success = 0;
            printf("TEST FAILED: col2im device kernel...\n");
            break;
        }
    }

    if (success) {
        printf("TEST PASSED: col2im device kernel!\n");
    }
#endif

    // CLEAN UP
    cudaFree(devA);
    cudaFree(devAc);
    
    free(matA);
#ifdef FUNCTEST
    free(matAc);
#endif
    free(retA);
    free(retAc);
}

int main()
{
    // Enforce default block and grid sizes
    unsigned int blockSize = 256;
    unsigned int gridSize = 0;

    // Calculate max needed kernels/threads number
    const unsigned int L = H - (K - 1);
    const unsigned int M = W - (K - 1);
    const unsigned int KERNELS_NUM = L * M * C;

    // Prepare variables for time measurement
    struct timeval t1, t2;
    double elapsedTime, totalTime = 0;
    int totalRuns = 1;
    
    // First warm-up run
    LOG("--------- WARM-UP ---------\n");
    program(256);
    LOG("--------- WARM-UP ---------\n\n");

#ifdef PERFTEST
    // Average over 10 runs
    totalRuns = 10;
    
    // Open file for perf logs
    std::fstream fperflog("perflog.csv", std::ios::out);
    if (fperflog.good())
    {
        // Measure effect of different block sizes
        const unsigned int MAX_BLOCK_SIZE = 2048;
        for (blockSize = 32; blockSize <= 34; blockSize *= 2) {
            const unsigned int MAX_GRID_SIZE = (KERNELS_NUM + blockSize - 1) / blockSize;
            LOG("  [!] For %d blocks, max grid size is %d\n", blockSize, MAX_GRID_SIZE);
            for (gridSize = 1; gridSize <= 1; gridSize *= 2) {
                if (gridSize <= MAX_GRID_SIZE) {
                    totalTime = 0;
                    for (int i = 0; i < 1; i++)
#endif
                    {
                        // Start timer
                        gettimeofday(&t1, NULL);
                    
                        // WORK HARD!
                        program(blockSize, gridSize);
                    
                        // Stop timer
                        gettimeofday(&t2, NULL);
                    
                        // Compute the elapsed time in millisec
                        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
                        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
                    
                        totalTime += elapsedTime;
                    }
                    LOG("  [!] Whole program took %.3fms averaged over %d runs\n", totalTime / totalRuns, totalRuns);
#ifdef PERFTEST
                    fperflog << blockSize << "," << gridSize << "," << elapsedTime << std::endl;
                } else {
                    // Meaningless data, there is more grids ten data cat utilize 
                    fperflog << blockSize << "," << gridSize << "," << -1 << std::endl;
                }
            }
        }
        
        // Close file
        fperflog.close();
    }
#endif

    return EXIT_SUCCESS;
}
