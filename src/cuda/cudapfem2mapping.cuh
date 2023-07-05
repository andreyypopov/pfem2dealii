#ifndef CUDAPFEM2MAPPING_H
#define CUDAPFEM2MAPPING_H

#include "cudapfem2finiteelement.cuh"

namespace cudaPfem2Mapping {
	template<int dim>
    __device__ void transform_local_to_global(double *global_pos, const double local_pos[dim], const double *vertices);

    template<>
    __device__ void transform_local_to_global<2>(double *global_pos, const double local_pos[2], const double *vertices);

    template<>
    __device__ void transform_local_to_global<3>(double *global_pos, const double local_pos[3], const double *vertices);
}

#endif // CUDAPFEM2MAPPING_H
