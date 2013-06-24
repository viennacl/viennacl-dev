__kernel void update_qr_column(__global float* A,
                               uint stride,
                               __global float* buf,
                               int m,
                               int n,
                               int last_n
                              )
{
    uint glb_id = get_global_id(0);
    uint glb_sz = get_global_size(0);

    for (int i = glb_id; i < last_n; i += glb_sz)
    {
        float a_ik = A[m * stride + i], a_ik_1, a_ik_2;

        a_ik_1 = A[(m + 1) * stride + i];

        for(int k = m; k < n; k++)
        {
            bool notlast = (k != n - 1);

            float p = buf[5 * k] * a_ik + buf[5 * k + 1] * a_ik_1;

            if (notlast)
            {
                a_ik_2 = A[(k + 2) * stride + i];
                p = p + buf[5 * k + 2] * a_ik_2;
                a_ik_2 = a_ik_2 - p * buf[5 * k + 4];
            }

            A[k * stride + i] = a_ik - p;
            a_ik_1 = a_ik_1 - p * buf[5 * k + 3];

            a_ik = a_ik_1;
            a_ik_1 = a_ik_2;
        }

        A[n * stride + i] = a_ik;
    }

}
