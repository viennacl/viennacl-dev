

// probably, this is a ugly way
__kernel void copy_col(__global float* A,
                       __global float* V,
                       uint row_start,
                       uint col_start,
                       uint size,
                       uint stride
                       ) {
    uint glb_id = get_global_id(0);
    uint glb_sz = get_global_size(0);

    for(uint i = row_start + glb_id; i < size; i += glb_sz) {
        V[i - row_start] = A[i * stride + col_start];
    }
}
