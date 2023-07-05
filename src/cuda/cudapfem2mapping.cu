#include "cudapfem2mapping.cuh"

namespace cudaPfem2Mapping {
    template<>
    __device__ void transform_local_to_global<2>(double *global_pos, const double local_pos[2], const double *vertices){
        double res[2];

        double x0 = 1.0 - local_pos[0];
        double x1 = local_pos[0];

        double v1[2], v2[2];
        v1[0] = *vertices;
        v1[1] = *(vertices + 1);
        v2[0] = *(vertices + 2);
        v2[1] = *(vertices + 3);
        for(int i = 0; i < 2; ++i)
            res[i] = (1.0 - local_pos[1]) * (x0 * v1[i] + x1 * v2[i]);

        v1[0] = *(vertices + 4);
        v1[1] = *(vertices + 5);
        v2[0] = *(vertices + 6);
        v2[1] = *(vertices + 7);
        for(int i = 0; i < 2; ++i)
            res[i] += local_pos[1] * (x0 * v1[i] + x1 * v2[i]);

        *global_pos = res[0];
        *(global_pos + 1) = res[1];
    }

    template<>
    __device__ void transform_local_to_global<3>(double *global_pos, const double local_pos[3], const double *vertices){
        double res[3];

        double x0 = 1.0 - local_pos[0];
        double x1 = local_pos[0];
        double y0 = 1.0 - local_pos[1];
        double y1 = local_pos[1];

        double v1[3], v2[3], tmp[3];
        for(int i = 0; i < 3; ++i){
            v1[i] = *(vertices + i);
            v2[i] = *(vertices + 3 + i);
            tmp[i] = y0 * (x0 * v1[i] + x1 * v2[i]);
        }

        for(int i = 0; i < 3; ++i){
            v1[i] = *(vertices + 6 + i);
            v2[i] = *(vertices + 9 + i);
            tmp[i] += y1 * (x0 * v1[i] + x1 * v2[i]);
            res[i] = (1.0 - local_pos[2]) * tmp[i];
        }

        for(int i = 0; i < 3; ++i){
            v1[i] = *(vertices + 12 + i);
            v2[i] = *(vertices + 15 + i);
            tmp[i] = y0 * (x0 * v1[i] + x1 * v2[i]);
        }

        for(int i = 0; i < 3; ++i){
            v1[i] = *(vertices + 18 + i);
            v2[i] = *(vertices + 21 + i);
            tmp[i] += y1 * (x0 * v1[i] + x1 * v2[i]);
            res[i] += local_pos[2] * tmp[i];
        }

        for(int i = 0; i < 3; ++i)
            *(global_pos + i) = res[i];
    }
}
