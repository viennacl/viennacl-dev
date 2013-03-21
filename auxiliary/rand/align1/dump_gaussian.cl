
// Dump random gaussian variables into the vector
__kernel void dump_gaussian(__global float *data, uint start, uint size, float mu, float sigma, uint baseOffset)
{
    unsigned int i = 2*get_global_id(0);
    if(i<size){
        const ulong MAX = 0xFFFF;
        mwc64xvec2_state_t rng;
        MWC64XVEC2_SeedStreams(&rng, baseOffset, 2);
        __global float *dest=data+start;
        uint2 x=MWC64XVEC2_NextUint2(&rng);
        float val1 = (x.s0 & MAX) / (float)MAX;
        float val2 = (x.s1 & MAX) / (float)MAX;
        float z1 = sqrt(-2*log(val1))*cos(2*M_PI*val2);
        float z2 = sqrt(-2*log(val1))*sin(2*M_PI*val2);
        dest[i] = mu + sigma*z1;
        if(i+1<size) dest[i+1] = mu + sigma*z2;
    }
}
