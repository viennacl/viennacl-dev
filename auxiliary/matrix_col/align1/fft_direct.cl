// Direct FFT computation (quadratic complexity - use for reference only)
__kernel void fft_direct(__global float2* input,
                         __global float2* output,
                         unsigned int size,
                         unsigned int stride,
                         unsigned int batch_num,
                         float sign) {

    const float NUM_PI = 3.14159265358979323846;

    for(unsigned int batch_id = 0; batch_id < batch_num; batch_id++) {
        for(unsigned int k = get_global_id(0); k < size; k += get_global_size(0)) {
            float2 f = 0.0f;

            for(unsigned int n = 0; n < size; n++) {
                float2 in = input[n * stride + batch_id]; //input index here

                float sn, cs;
                float arg = sign * 2 * NUM_PI * k / size * n;
                sn = sincos(arg, &cs);

                float2 ex = (float2)(cs, sn);
                f = f + (float2)(in.x * ex.x - in.y * ex.y, in.x * ex.y + in.y * ex.x);
            }

            output[k * stride + batch_id] = f;// output index here
        }
    }
}
