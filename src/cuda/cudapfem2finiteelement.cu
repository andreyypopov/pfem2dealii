#include "cudapfem2finiteelement.cuh"

#include "math_constants.h"

namespace cudaPfem2FiniteElement {
    template<>
    __device__ double shape_value<2>(const unsigned int i, const double p[2]){
        switch (i)
        {
        case 0:
            return (1.0 - p[0]) * (1.0 - p[1]);
        case 1:
            return p[0] * (1.0 - p[1]);
        case 2:
            return (1.0 - p[0]) * p[1];
        case 3:
            return p[0] * p[1];
        default:
            return CUDART_NAN;
        }
    }

    template<>
    __device__ double shape_value<3>(const unsigned int i, const double p[3]){
        switch (i)
        {
        case 0:
            return (1 - p[0]) * (1 - p[1]) * (1 - p[2]);
        case 1:
            return p[0] * (1 - p[1]) * (1 - p[2]);
        case 2:
            return (1 - p[0]) * p[1] * (1 - p[2]);
        case 3:
            return p[0] * p[1] * (1 - p[2]);
        case 4:
            return (1 - p[0]) * (1 - p[1]) * p[2];
        case 5:
            return p[0] * (1 - p[1]) * p[2];
        case 6:
            return (1 - p[0]) * p[1] * p[2];
        case 7:
            return p[0] * p[1] * p[2];
        default:
            return CUDART_NAN;
        }
    }
}
