#include "cudapfem2particlehandler.cuh"

#include <deal.II/base/timer.h>

#include "../pfem2solver.h"
#include "cuda_helper.cuh"
#include "cudapfem2fem.cuh"
#include "cudapfem2mapping.cuh"

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
				localPosition[i] = (cellPartsIndices[i] + 0.5) * h[i];

			cudaPfem2Mapping::transform_local_to_global<dim>((double*)&globalPosition, localPosition, cell->get_vertex_coords());
			cudaPfem2Particle<dim> particle((double*)&globalPosition, (double*)&localPosition, particleID++);
			particle.cell = cell;

			*(cellParticlesStart + j) = particle;
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
	cudaFree(d_quantities);
	cudaFree(d_cellPartsIndices);
}

template<int dim>
void cudaPfem2ParticleHandler<dim>::seed_particles()
{
    //generate possible combinations of indices for cell parts within a single cell
    unsigned int cellPartsCount = this->fill_cell_parts_indices();

    checkCudaErrors(cudaMemcpyToSymbol(d_quantities, &this->quantities, sizeof(unsigned int) * dim, 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_cellPartsCount, &cellPartsCount, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_cellPartsIndices, sizeof(unsigned int) * dim * cellPartsCount));
	checkCudaErrors(cudaMemcpy(d_cellPartsIndices, this->cellPartsIndices.data(), sizeof(unsigned int) * dim * cellPartsCount, cudaMemcpyHostToDevice));

	int n_cells = this->mainSolver->getTriangulation().n_cells();
	checkCudaErrors(cudaMalloc(&d_particles, sizeof(cudaPfem2Particle<dim>) * cellPartsCount * n_cells));

	unsigned int blocks = blocksForSize(n_cells);

	const auto femSolver = static_cast<cudaPfem2Fem<dim>*>(&this->mainSolver->getFemSolver());
	kernelSeedParticlesIntoCell<dim><<<blocks, gpuThreads>>>(d_particles, femSolver->getCells(), d_cellPartsIndices, n_cells);

	pfem2ParticleHandler<dim>::seed_particles();
}
