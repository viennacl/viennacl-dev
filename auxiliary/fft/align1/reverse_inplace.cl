// reverses the entries in a vector
__kernel void reverse_inplace(__global float* vec, uint size) {
    for(uint i = get_global_id(0); i < (size >> 1); i+=get_global_size(0)) {
        float val1 = vec[i];
        float val2 = vec[size - i - 1];

        vec[i] = val2;
        vec[size - i - 1] = val1;
    }
}

