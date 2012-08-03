

__kernel void bidiag_pack(__global float* A,
                          __global float* D,
                          __global float* S,
                          uint size1,
                          uint size2,
                          uint stride
                          ) {
    uint size = min(size1, size2);

    if(get_global_id(0) == 0)
        S[0] = 0;

    for(uint i = get_global_id(0); i < size ; i += get_global_size(0)) {
        D[i] = A[i*stride + i];
        S[i + 1] = (i + 1 < size2) ? A[i*stride + (i + 1)] : 0;
    }
}
