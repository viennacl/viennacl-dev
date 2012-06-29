

// probably, this is too
__kernel void copy_row(__global float* A,
                       __global float* V,
                       uint row_start,
                       uint col_start,
                       uint size,
                       uint stride
                       ) {
    uint glb_id = get_global_id(0);
    uint glb_sz = get_global_size(0);

    for(uint i = col_start + glb_id; i < size; i += glb_sz) {
        V[i - col_start] = A[row_start * stride + i];
    }
}
