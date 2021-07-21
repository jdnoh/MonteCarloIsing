// variable type for spin related variables can be set here
typedef signed char spin_t;
// typedef int spin_t;

#include "subIsing_AB.c"

#define MaxThreads (128)

int main(int argc, char *argv[])
{
    if(argc!=4) error_output("command Lsize tmax delt");
    const int Lsize = atoi(argv[1]); // input of lattice size
    const int tmax = atoi(argv[2]);  // input of simulation time
    const int delt = atoi(argv[3]);  // input of measurement interval

    // square lattice with nSites = Lsize*Lsize
    // whole lattice = sublattices A(even) + B(odd)
    // |sublattice| = Lx * Ly with Lx = Lsize/2 and Ly = Lsize 
    const int nSites = Lsize*Lsize; 
    const int Lx = Lsize/2;
    const int Ly = Lsize ;
    const int nSitesHalf = nSites/2;

    // grid configuration
    // nThread = min[ Lx, MaxThreads], nBlocks = Ly
    const int nThreads = (Lx < MaxThreads)? Lx : MaxThreads;
    const int nBlocks = nSitesHalf / nThreads;

    // seeds for CPU and GPU PRNG
    unsigned int seedCPU, seedGPU;
    // get_PRNG_seeds(&seedCPU, &seedGPU);
    seedCPU = 1234; seedGPU = 1234;
    printf("# seedCPU = %u seedGPU = %u\n", seedCPU, seedGPU);

    // CPU PRNG setup
    // Mersenne-Twrister engine initialized with seedCPU
    std::mt19937 cpuPRNG(seedCPU);
    // random number distribution type: uniform in [0,1)
    std::uniform_real_distribution<float> rand_uniform(0.0, 1.0);
    // usage: r = rand_uniform(cpuPRNG)

    // GPU PRNG setup
    curandGenerator_t gen;
    // default (WOWXOR) engine
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // initialization with the seedGPU
    curandSetPseudoRandomGeneratorSeed(gen, seedGPU);

    // device memory for spins, local magnetization, and energy
    // variable type: "int" (4 byte) or "signed char" (1 byte)
    // spin_t = int or signed char 
    spin_t *devSpinA, *devSpinB, *devMaglocal, *devElocal;

    // memory allocation
    prepare1_memory(&devSpinA, &devSpinB, &devMaglocal, 
            &devElocal, nSitesHalf);

    // memory for random numbers and Boltzmann factors
    float *devRand, *devBoltzmannFactor;
    prepare2_memory(&devBoltzmannFactor, &devRand, nSitesHalf);
    
    float beta = log(1.+sqrt(2.))/2.0;
    // for(float beta=0.3; beta<0.6; beta += 0.02) {
        // setup of Boltzmann factor e^{-beta dE} 
        set_weight(devBoltzmannFactor, beta);

        // random initial spin configuration
        curandGenerateUniform(gen, devRand, nSitesHalf);
        init_rand_spins<<<nBlocks, nThreads>>>(devSpinA, devRand);
        curandGenerateUniform(gen, devRand, nSitesHalf);
        init_rand_spins<<<nBlocks, nThreads>>>(devSpinB, devRand);

        printf("# Lsize beta t Energy Magnetization\n");
        for(int t=1; t<=tmax; t++) {
            // random numbers generation 
            curandGenerateUniform(gen, devRand, nSitesHalf);

            // random sublattice update
            if(rand_uniform(cpuPRNG)<0.5) 
                // A sublattice update
                spinFlip_AB<<<nBlocks, nThreads>>>(0, Lx, Ly, devSpinA,
                    devSpinB, devBoltzmannFactor, devRand);
            else
                // or B sublattice update
                spinFlip_AB<<<nBlocks, nThreads>>>(1, Lx, Ly, devSpinB,
                    devSpinA, devBoltzmannFactor, devRand);

            if(t%delt==0) {  // measurement
                long int magnetization, energy;
                get_orderParameter(Lx, Ly, nSitesHalf, devSpinA, 
                        devSpinB, devMaglocal, devElocal,
                        &magnetization, &energy,
                        nBlocks, nThreads);
                // output 
                printf("%8d %18e %12d %12ld %12ld\n", Lsize, beta, t, 
                        energy, magnetization);
            }

            // thread synchronization
            // not necessary, but just for the peace of mind
            cudaDeviceSynchronize(); 
        }
    //}

    cudaFree(devBoltzmannFactor); cudaFree(devRand); 
    cudaFree(devSpinA); cudaFree(devSpinB), cudaFree(devElocal);
}
