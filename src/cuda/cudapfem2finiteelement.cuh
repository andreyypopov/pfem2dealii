#ifndef CUDAPFEM2FINITEELEMENT_H
#define CUDAPFEM2FINITEELEMENT_H

#include <deal.II/base/point.h>

using namespace dealii;

namespace cudaPfem2FiniteElement {
    template<int dim>
    __device__ double shape_value(const unsigned int i, const double p[dim]);

    template<>
    __device__ double shape_value<2>(const unsigned int i, const double p[2]);

    template<>
    __device__ double shape_value<3>(const unsigned int i, const double p[3]);
}

#endif // CUDAPFEM2FINITEELEMENT_H
