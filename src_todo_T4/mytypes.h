#ifndef MYTYPES_H
#define MYTYPES_H

// tilescale (# of points computed by each thread)
#ifndef TILESCALE_M
#define TILESCALE_M 8 // Enter your own values
#endif
#ifndef TILESCALE_N
#define TILESCALE_N 8 // Enter your own values
#endif
#ifndef TILESCALE_K
#define TILESCALE_K 4 // Enter your own values
#endif

#define TILEDIM_M (BLOCKDIM_Y*TILESCALE_M) // Enter your own values
#define TILEDIM_N (BLOCKDIM_X*TILESCALE_N) // Enter your own values

// matrix A loads
// with warps along the horiziontal axis (K)
// so to get good coalescaed loads, we want TILEDIM_K to be >= 32
//
#define TILEDIM_K 64 // Enter your own values

// step size in each dimension
#define TILESTEP_N (TILEDIM_N/TILESCALE_N) // Enter your own values
#define TILESTEP_K (TILEDIM_K/TILESCALE_K) // Enter your own values
#define TILESTEP_M (TILEDIM_M/TILESCALE_M) // Enter your own values
 
#endif
