#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <cuda.h>
#include <random>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

const int nNeighbor = 4;

void error_output(const char *desc)
{
    printf("%s\n", desc) ; exit(-1) ;
}

// random sublattice spin flips using the Metropolis algorithm
// A (B) sublattice update when even_odd = 0 (1)
// neighbors of an A site (x,y) = (x,y+1), (x,y-1), (x-(-1)^y,y), (x,y) 
// neighbors of a  B site (x,y) = (x,y+1), (x,y-1), (x+(-1)^y,y), (x,y)
__global__ void spinFlip_AB(int even_odd, const int lx, const int ly, 
        spin_t *spinActive, const spin_t* spinInact, 
        const float* dev_bw, const float* rn)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    // coordinates in the 2D (Lx * Ly) lattice 
    const int x = tid%lx; 
    const int y = tid/lx;

    const spin_t s0 = spinActive[tid]; // my spin states
    int nSame=0 ;  // number of neighbors of the same state
    int site;

    // interaction with horizontal neighbors
    if(s0==spinInact[tid]) nSame ++;

    if((y%2) == even_odd) {  // neighbor at x' = x-1
        site = (x==0)? tid+lx-1 : tid-1;
    } else {                 // neighbor at x' = x+1
        site = (x==(lx-1))? tid+1-lx : tid+1;
    }
    if(s0==spinInact[site]) nSame ++;

    // upper
    site = (y==(ly-1)) ? tid-lx*(ly-1) : tid+lx;
    if(s0==spinInact[site]) nSame ++;

    // below
    site = (y==0)? tid+lx*(ly-1) : tid-lx;
    if(s0==spinInact[site]) nSame ++;

     // Metropolis algorithm
     if(rn[tid]<=dev_bw[nSame]) spinActive[tid] = -s0;
}

// random spin initialization
__global__ void init_rand_spins(spin_t *spin, const float* rand)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(rand[tid]>0.5) spin[tid] = 1;
    else spin[tid] = -1;
}

// local energy and mamgetization in the perspective of A sublattice
__global__ void mag_energy_local(int lx, int ly, 
        const spin_t *spinA, const spin_t *spinB, 
        spin_t *maglocal, spin_t *elocal)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int x=tid%lx;
    int y=tid/lx;

    int en, site;

    // horizontal bonds
    en = spinB[tid];
    if(y%2==0) { // neighbor at x-1
        site = (x==0)? tid+lx-1 : tid-1;
    } else {     // neighbor at x+1
        site = (x==lx-1)? tid+1-lx : tid+1;
    }
    en += spinB[site];

    // upper
    site = (y==ly-1)? tid-lx*(ly-1) : tid + lx;
    en += spinB[site];

    // bottom
    site = (y==0)? tid+lx*(ly-1) : tid - lx;
    en += spinB[site];

    elocal[tid]   = -en*spinA[tid];
    maglocal[tid] = spinA[tid] + spinB[tid];
}

void get_orderParameter(const int lx, const int ly, 
       const int nSiteHalf, const spin_t *spinA,
       const spin_t *spinB, spin_t *localMag, spin_t *localEn,
       long int *mag, long int *en,
       const int nBlocks, const int nThreads)
{
    mag_energy_local<<<nBlocks, nThreads>>>(lx, ly, spinA, 
            spinB, localMag, localEn);

    *mag = thrust::reduce(thrust::device_ptr<spin_t>(localMag), 
            thrust::device_ptr<spin_t>(localMag)+nSiteHalf,0,
            thrust::plus<long int>());
    *en  = thrust::reduce(thrust::device_ptr<spin_t>(localEn), 
            thrust::device_ptr<spin_t>(localEn)+nSiteHalf,0,
            thrust::plus<long int>());
}

void set_weight(float *dev_bw, float beta)
{
    // bw[n] : spin flip prob. when there are n ferromagnetic neighbors
    // Ei = -n + (Z-n) = Z-2n
    // Ef = n - (Z-n) = 2n-Z
    // bw[n] = exp[-beta(Ef-Ei)] = exp[2*beta*(Z-2n)]
    
    float *bw;
    bw = (float *)malloc((nNeighbor+1)*sizeof(float));
    
    for(int n=0; n<=nNeighbor; n++) 
        bw[n] = exp(2.*beta*(nNeighbor-2*n));
    cudaMemcpy(dev_bw, bw, sizeof(float)*(nNeighbor+1), cudaMemcpyHostToDevice);
    free(bw);
}

// generate two PRNG seeds, one for CPU and another for GPU PRNG
// they are generated from the std::random_device
void get_PRNG_seeds(unsigned int* seed0, unsigned int* seed1)
{
    std::random_device rd;
    *seed0 = rd();
    *seed1 = rd();
    return;
}

void prepare1_memory(spin_t **a, spin_t **b, spin_t **c, spin_t **d, int n)
{
    cudaMalloc(a, sizeof(spin_t)*n);
    cudaMalloc(b, sizeof(spin_t)*n);
    cudaMalloc(c, sizeof(spin_t)*n);
    cudaMalloc(d, sizeof(spin_t)*n);
}

void prepare2_memory(float **a, float **b, int n)
{
    cudaMalloc(a, sizeof(float)*n);
    cudaMalloc(b, sizeof(float)*n);
}
