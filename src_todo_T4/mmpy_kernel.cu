// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}

#else
extern __shared__ _FTYPE_ sharmem[];

//You should be changing the kernel here for the non naive implementation.
__global__ void matMul(int N, _FTYPE_ * __restrict__ C, _FTYPE_ * __restrict__ A, _FTYPE_ * __restrict__ B) {
	//__shared__ _FTYPE_ As[TILEDIM_M][TILEDIM_K], Bs[TILEDIM_K][TILEDIM_N];
	_FTYPE_ * __restrict__ As = &sharmem[0];
	_FTYPE_ * __restrict__ Bs = &As[TILEDIM_M * TILEDIM_K];
	
	int ty = threadIdx.y, tx = threadIdx.x;
	int by = blockIdx.y, bx = blockIdx.x;
	
    	int I =  blockIdx.y*TILEDIM_M + threadIdx.y;
    	int J =  blockIdx.x*TILEDIM_N + threadIdx.x;
	
	register _FTYPE_ Cij[TILESCALE_M][TILESCALE_N] = {0};
	
	int kk = 0;
	int numTiles = N/TILEDIM_K;
	if(N % TILEDIM_K != 0)
		numTiles++;

	#pragma unroll
	for(kk=0; kk<numTiles; kk++)
	{
		// Assuming that TILEDIM is same in m, n directions
		#pragma unroll
		for(int row=0; row<TILEDIM_M; row+=TILESTEP_M)
		{
			#pragma unroll
			for(int col=0; col<TILEDIM_K; col+=TILESTEP_K)
			{
				As[(ty+row)*TILEDIM_K + tx + col] = A[(I+row)*N + kk*TILEDIM_K + tx + col];
			}
		}

		#pragma unroll
		for(int row=0; row<TILEDIM_K; row+=TILESTEP_K)
		{
			#pragma unroll
			for(int col=0; col<TILEDIM_N; col+=TILESTEP_N)
			{
				Bs[(ty+row)*TILEDIM_N + tx + col] = B[(kk*TILEDIM_K+ty+row)*N + J + col];
			}
		}
		
		__syncthreads();
		
		#pragma unroll
		for(int k=0; k<TILEDIM_K; k++)
		{
			#pragma unroll
			for(int row=0; row<TILEDIM_M; row+=TILESTEP_M)
			{
				#pragma unroll
				for(int col=0; col<TILEDIM_N; col+=TILESTEP_N)
				{
						Cij[row/TILESTEP_M][col/TILESTEP_N] += As[(ty+row)*TILEDIM_K + k]*Bs[k*TILEDIM_N + tx + col];
				}
			}
		}
	
		__syncthreads();
	}
	
	#pragma unroll
	for(int row=0; row<TILEDIM_M; row+=TILESTEP_M)
	{
		#pragma unroll
		for(int col=0; col<TILEDIM_N; col+=TILESTEP_N)
		{
			if(((I+row)<N)&&((J+col)<N)){
				C[(I+row)*N + J + col] = Cij[row/TILESTEP_M][col/TILESTEP_N];
			}
		}
	}	
}
#endif
