// computes the matrix vector product with a Vandermonde matrix
__kernel void vandermonde_prod(__global float* vander,
                                __global float* vector,
                                __global float* result,
                                uint size) {
    for(uint i = get_global_id(0); i < size; i+= get_global_size(0)) {
        float mul = vander[i];
        float pwr = 1;
        float val = 0;

        for(uint j = 0; j < size; j++) {
            val = val + pwr * vector[j];
            pwr *= mul;
        }

        result[i] = val;
    }
}

