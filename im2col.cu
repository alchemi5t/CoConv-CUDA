#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

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

const unsigned int H = 4, W = 4, C = 1, K = 2; 

// HOST FUNCTION
// Takes matrix A [float *matA] and transforms it
// into column representation [float *matAc]

void flatten_kernel(float * canvas, float * weights, int k, int d, int c_rows){
	int c_cols = k + (k-1)*(d-1);
	itr = 0;
	for(int dilation = 1; dilation<=d; dilation ++){
		int cur_kernel_size = k + (k-1)*(dilation-1);
		for(int kernel_id = 0; kernel_id <c_rows/4; kernel_id++){
			for(int weight_id = 0; weight_id < k*k*C; elem++){
				canvas[itr] = weights[weight_id];
				itr++;
				if((weight_id+1)%(k)==0){
					for(int last_col_pads = 0; last_col_pads<(dilation-1)(c_cols) + (c_cols-cur_kernel_size);last_col_pads++ ){
						canvas[itr] = 0;
						itr++;
					}
				}
				else{
					
					for(int inner_cols = 0; inner_cols<(dilation-1);inner_cols++ ){
						canvas[itr] = 0;
						itr++;
					}
				
				}
		
			}
		}
	
	
	
	
	
	}
	
	
	
}


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
        int m = (idx/C ) / L;
        int l = (idx/C ) % L;
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
                            matAc[(idx*K*K) + (x*K + y)] = matA[(r*(H*W)+(w*W + h ))]; 
			h++;                        
}
                    }w++;
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
        int m = (idx/C ) / L;
        int l = (idx/C ) % L;
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
                           matA[(r*(H*W)+(w*W + h ))]  = matAc[(idx*K*K) + (x*K + y)]; 
			h++;                        
}
                    }w++;
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
	if((i+1)%(C*K*K)==0)
{printf("\n");}
    }
printf("\n");

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
