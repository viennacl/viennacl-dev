
// Dump uniform random variable into the given vector
__kernel void dump_uniform(__global float *data, uint start, uint size, float a, float b, uint baseOffset)
{
      unsigned int i = get_global_id(0);
      if(i<size){
          const ulong MAX = 0xFFFF;
          mwc64x_state_t rng;
          MWC64X_SeedStreams(&rng, baseOffset, 1);
          __global float *dest=data+start;
          uint x=MWC64X_NextUint(&rng);
          float val = (x&MAX)/(float)MAX;
          float s = (b-a);
          dest[i]= a + s*val;
      }
}
