#ifndef CUDAPFEM2PARTICLEHANDLER_H
#define CUDAPFEM2PARTICLEHANDLER_H

#include "../pfem2particlehandler.h"
#include "cudapfem2fem.cuh"
#include "cudapfem2particle.cuh"

__constant__ unsigned int d_quantities[3];
__constant__ unsigned int d_cellPartsCount;

__constant__ unsigned int d_ndofs;
__constant__ double d_timestep;

using namespace dealii;

template<int dim>
__global__ void kernelSeedParticlesIntoCell(cudaPfem2Particle<dim> *particles, const cudaPfem2Cell<dim> *cells,
	const unsigned int *cellPartsIndices, int n);

template<int dim>
__global__ void kernelCorrectParticleVelocity(cudaPfem2Particle<dim> *particles, const double *solutionV,
	const double *oldSolutionV, int n);

template<int dim>
__global__ void kernelTransferParticles(cudaPfem2Particle<dim> *particles, const double *solutionV, int n);

template<int dim>
__global__ void kernelPrepareProjection(double *projectedVelocity, double *projectedWeights, int n);

template<int dim>
__global__ void kernelProjectParticleVelocity(double *projectedVelocity, double *projectedWeights, const double value, const cudaPfem2Particle<dim> *particles, int n);

template<int dim>
__global__ void kernelUpdateNodeVelocity(double *solutionV, const double *projectedVelocity, const double *projectedWeights, int n);

template<int dim>
class cudaPfem2ParticleHandler : public pfem2ParticleHandler<dim>
{
public:
	cudaPfem2ParticleHandler(const FE_Q<dim> *finite_element);
	virtual ~cudaPfem2ParticleHandler();

	virtual void seed_particles() override;
	virtual void correct_particle_velocity() override;
	virtual void project_particle_fields() override;

private:
	cudaPfem2Particle<dim> *d_particles;

	double *d_projectedVelocity;
	double *d_projectedWeights;

	cudaPfem2Fem<dim> *femSolver;

	unsigned int *d_cellPartsIndices;

	unsigned int currentParticleCount;
};

template class cudaPfem2ParticleHandler<2>;

template class cudaPfem2ParticleHandler<3>;

#endif // CUDAPFEM2PARTICLEHANDLER_H
