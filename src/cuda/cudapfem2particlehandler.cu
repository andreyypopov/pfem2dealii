#include "cudapfem2particlehandler.cuh"

#include <deal.II/base/timer.h>

#include "../pfem2solver.h"
#include "cuda_helper.cuh"
#include "cudapfem2fem.cuh"
#include "cudapfem2mapping.cuh"
#include "cudapfem2finiteelement.cuh"

template<int dim>
__global__ void kernelSeedParticlesIntoCell(cudaPfem2Particle<dim> *particles, const cudaPfem2Cell<dim> *cells,
	const unsigned int *cellPartsIndices, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		const cudaPfem2Cell<dim> *cell = cells + i;

		double h[dim];
		for(int j = 0; j < dim; ++j)
			h[j] = 1.0 / d_quantities[j];

		cudaPfem2Particle<dim> *cellParticlesStart = particles + i * d_cellPartsCount;
		int particleID = i * d_cellPartsCount;

		double localPosition[dim], globalPosition[dim];

		for(int j = 0; j < d_cellPartsCount; ++j){
			for(int k = 0; k < dim; ++k)
				localPosition[k] = (cellPartsIndices[k] + 0.5) * h[k];

			cudaPfem2Mapping::transform_local_to_global<dim>((double*)&globalPosition, localPosition, cell->get_vertex_coords());
			cudaPfem2Particle<dim> particle((double*)&globalPosition, (double*)&localPosition, particleID++);
			particle.cell = cell;

			*(cellParticlesStart + j) = particle;
		}
	}
}

template<int dim>
__global__ void kernelCorrectParticleVelocity(cudaPfem2Particle<dim> *particles, const double *solutionV, const double *oldSolutionV, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		cudaPfem2Particle<dim> *particle = particles + i;

		double deltaV[dim]{0.0};
		double shapeValue;
		types::global_dof_index jDofIndex;

		for(int j = 0; j < GeometryInfo<dim>::vertices_per_cell; ++j){
			shapeValue = cudaPfem2FiniteElement::shape_value<dim>(j, particle->reference_location);
			jDofIndex = particle->cell->get_dof_indices()[j];

			for(int k = 0; k < dim; ++k)
				deltaV[k] += shapeValue * (solutionV[jDofIndex + k * d_ndofs] - ((oldSolutionV) ? oldSolutionV[jDofIndex + k * d_ndofs] : 0.0));
		}

		for(int k = 0; k < dim; ++k)
			particle->velocity[k] += deltaV[k];
	}
}

template<int dim>
__global__ void kernelTransferParticles(cudaPfem2Particle<dim> *particles, const double *solutionV, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		cudaPfem2Particle<dim> *particle = particles + i;

		double vel[dim]{0.0};
		double shapeValue;
		types::global_dof_index jDofIndex;

		for(int j = 0; j < GeometryInfo<dim>::vertices_per_cell; ++j){
			shapeValue = cudaPfem2FiniteElement::shape_value<dim>(j, particle->reference_location);
			jDofIndex = particle->cell->get_dof_indices()[j];

			for(int k = 0; k < dim; ++k)
				vel[k] += shapeValue * solutionV[jDofIndex + k * d_ndofs];
		}

		for(int k = 0; k < dim; ++k){
			particle->location[k] += d_timestep * vel[k];
			particle->velocity_ext[k] = vel[k];
		}
	}
}

template<int dim>
__global__ void kernelPrepareProjection(double *projectedVelocity, double *projectedWeights, const double value, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		for(int k = 0; k < dim; ++k)
			projectedVelocity[i + k * d_ndofs] = value;

		projectedWeights[i] = value;
	}
}

template<int dim>
__global__ void kernelProjectParticleVelocity(double *projectedVelocity, double *projectedWeights, const cudaPfem2Particle<dim> *particles, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		const cudaPfem2Particle<dim> *particle = particles + i;

		double shapeValue;
		types::global_dof_index jDofIndex;

		for(int j = 0; j < GeometryInfo<dim>::vertices_per_cell; ++j){
			shapeValue = cudaPfem2FiniteElement::shape_value<dim>(j, particle->reference_location);
			jDofIndex = particle->cell->get_dof_indices()[j];

			for(int k = 0; k < dim; ++k)
				atomicAdd(&projectedVelocity[jDofIndex + k * d_ndofs], shapeValue * particle->velocity[k]);

			atomicAdd(&projectedWeights[jDofIndex], shapeValue);
		}
	}
}

template<int dim>
__global__ void kernelUpdateNodeVelocity(double *solutionV, const double *projectedVelocity, const double *projectedWeights, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		unsigned int dofNumber;
		for(int k = 0; k < dim; ++k){
			dofNumber = i + k * d_ndofs;
			solutionV[dofNumber] = projectedVelocity[dofNumber] / projectedWeights[i];
		}
	}
}

template<int dim>
cudaPfem2ParticleHandler<dim>::cudaPfem2ParticleHandler(const FE_Q<dim> *finite_element)
	: pfem2ParticleHandler<dim>(finite_element)
{

}

template<int dim>
cudaPfem2ParticleHandler<dim>::~cudaPfem2ParticleHandler()
{
	cudaFree(d_particles);
	cudaFree(d_projectedVelocity);
	cudaFree(d_projectedWeights);
	cudaFree(d_quantities);
	cudaFree(d_cellPartsIndices);
}

template<int dim>
void cudaPfem2ParticleHandler<dim>::seed_particles()
{
	pfem2ParticleHandler<dim>::seed_particles();

    //generate possible combinations of indices for cell parts within a single cell
	const unsigned int cellPartsCount = this->fill_cell_parts_indices();

    checkCudaErrors(cudaMemcpyToSymbol(d_quantities, &this->quantities, sizeof(unsigned int) * dim, 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_cellPartsCount, &cellPartsCount, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_cellPartsIndices, sizeof(unsigned int) * dim * cellPartsCount));
	checkCudaErrors(cudaMemcpy(d_cellPartsIndices, this->cellPartsIndices.data(), sizeof(unsigned int) * dim * cellPartsCount, cudaMemcpyHostToDevice));

	int n_cells = this->mainSolver->getTriangulation().n_cells();
	currentParticleCount = cellPartsCount * n_cells;
	checkCudaErrors(cudaMalloc(&d_particles, sizeof(cudaPfem2Particle<dim>) * currentParticleCount));

	unsigned int blocks = blocksForSize(n_cells);

	this->femSolver = static_cast<cudaPfem2Fem<dim>*>(&this->mainSolver->getFemSolver());
	const unsigned int n_dofs = femSolver->getDoFhandler().n_dofs();
	checkCudaErrors(cudaMemcpyToSymbol(d_ndofs, &n_dofs, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
	kernelSeedParticlesIntoCell<dim><<<blocks, gpuThreads>>>(d_particles, femSolver->getCells(), d_cellPartsIndices, n_cells);
	getLastCudaError("Particle seeding");

	//call correct_particle_velocity() to initialize the particle velocity field
	blocks = blocksForSize(currentParticleCount);
	kernelCorrectParticleVelocity<dim><<<blocks, gpuThreads>>>(d_particles, femSolver->getDeviceSolutionV(), nullptr, currentParticleCount);
	getLastCudaError("Particle velocity field initialization");

	const double time_step = this->mainSolver->getParameterHandler().getTimeStep();
	checkCudaErrors(cudaMemcpyToSymbol(d_timestep, &time_step, sizeof(double), 0, cudaMemcpyHostToDevice));

	//prepare vectors for particle velocity projection
	checkCudaErrors(cudaMalloc(&d_projectedVelocity, sizeof(double) * n_dofs * dim));
	checkCudaErrors(cudaMalloc(&d_projectedWeights, sizeof(double) * n_dofs));
}

template <int dim>
void cudaPfem2ParticleHandler<dim>::correct_particle_velocity()
{
	pfem2ParticleHandler<dim>::correct_particle_velocity();

	unsigned int blocks = blocksForSize(currentParticleCount);
	kernelCorrectParticleVelocity<dim><<<blocks, gpuThreads>>>(d_particles, femSolver->getDeviceSolutionV(), femSolver->getDeviceOldSolutionV(), currentParticleCount);
	getLastCudaError("Particle velocity correction");
}

template <int dim>
void cudaPfem2ParticleHandler<dim>::project_particle_fields()
{
	const unsigned int n_dofs = this->femSolver->getDoFhandler().n_dofs();

	//set the projected velocity and weights vectors to zero
	unsigned int blocks = blocksForSize(n_dofs);
	kernelPrepareProjection<dim><<<blocks, gpuThreads>>>(d_projectedVelocity, d_projectedWeights, 0.0, n_dofs);
	getLastCudaError("Projection vectors preparation");

	//project the particle velocity to intermediate vectors
	blocks = blocksForSize(currentParticleCount);
	kernelProjectParticleVelocity<dim><<<blocks, gpuThreads>>>(d_projectedVelocity, d_projectedWeights, d_particles, currentParticleCount);
	getLastCudaError("Particle velocity projection");

	//update the node velocities
	blocks = blocksForSize(n_dofs);
	kernelUpdateNodeVelocity<dim><<<blocks, gpuThreads>>>(femSolver->getDeviceSolutionV(), d_projectedVelocity, d_projectedWeights, n_dofs);
	getLastCudaError("Node velocity update");

	pfem2ParticleHandler<dim>::project_particle_fields();
}
