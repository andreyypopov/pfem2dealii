#ifndef CUDAPFEM2PARTICLE_H
#define CUDAPFEM2PARTICLE_H

#include "cudapfem2fem.cuh"

template<int dim>
class cudaPfem2Particle
{
public:
	__host__ __device__ cudaPfem2Particle();
	__host__ __device__ cudaPfem2Particle(const double *location, const double *reference_location, const unsigned int id);

	double location[dim];
	double reference_location[dim];
	unsigned int id;

	const cudaPfem2Cell<dim> *cell;

	double velocity[dim];
	double velocity_ext[dim];
};

template class cudaPfem2Particle<2>;

template class cudaPfem2Particle<3>;

#endif // CUDAPFEM2PARTICLE_H
