#include "cudapfem2fem.cuh"

#include <deal.II/base/timer.h>

#include "../pfem2solver.h"
#include "cuda_helper.cuh"

template<int dim>
cudaPfem2Cell<dim>::cudaPfem2Cell(const int index)
	: index(index)
{
}

template <int dim>
void cudaPfem2Cell<dim>::set_vertex_index(const unsigned int vertex_index, int i)
{
	vertex_indices[i] = vertex_index;
}

template <int dim>
void cudaPfem2Cell<dim>::set_dof_index(const unsigned int dof_index, int i)
{
	dof_indices[i] = dof_index;
}

template <int dim>
void cudaPfem2Cell<dim>::set_vertex_coords(const Point<dim> &p, int i)
{
	for(int coord = 0; coord < dim; ++coord)
		vertex_coords[i * dim + coord] = p[coord];
}

template <int dim>
__host__ __device__ const double *cudaPfem2Cell<dim>::get_vertex_coords() const
{
    return vertex_coords;
}

template <int dim>
__host__ __device__ const types::global_dof_index *cudaPfem2Cell<dim>::get_dof_indices() const
{
    return dof_indices;
}

template<int dim>
cudaPfem2Fem<dim>::cudaPfem2Fem(const FE_Q<dim> *finite_element)
	: pfem2Fem<dim>(finite_element)
{

}

template<int dim>
cudaPfem2Fem<dim>::~cudaPfem2Fem()
{
	cudaFree(d_vertices);
	cudaFree(d_cells);
	cudaFree(d_solutionV);
	cudaFree(d_oldSolutionV);
}

template<int dim>
void cudaPfem2Fem<dim>::setup_system()
{
	pfem2Fem<dim>::setup_system();

	const auto& tria = this->mainSolver->getTriangulation();

	checkCudaErrors(cudaMalloc(&d_vertices, sizeof(Point<dim>) * tria.n_vertices()));
	checkCudaErrors(cudaMemcpy(d_vertices, tria.get_vertices().data(), sizeof(Point<dim>) * tria.n_vertices(), cudaMemcpyHostToDevice));

	std::vector<cudaPfem2Cell<dim>> hostCells;
	hostCells.reserve(tria.n_cells());

	for(const auto &cell : this->dof_handler.cell_iterators()){
		cudaPfem2Cell<dim> hostCell(cell->index());

		for(int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i){
			hostCell.set_vertex_index(cell->vertex_index(i), i);
			hostCell.set_dof_index(cell->vertex_dof_index(i, 0), i);
			hostCell.set_vertex_coords(cell->vertex(i), i);
		}

		hostCells.push_back(hostCell);
	}

	checkCudaErrors(cudaMalloc(&d_cells, sizeof(cudaPfem2Cell<dim>) * tria.n_cells()));
	checkCudaErrors(cudaMemcpy(d_cells, hostCells.data(), sizeof(cudaPfem2Cell<dim>) * tria.n_cells(), cudaMemcpyHostToDevice));

	for(int i = 0; i < dim; ++i){
		hostSolutionV[i].reinit(this->n_dofs);
		hostOldSolutionV[i].reinit(this->n_dofs);
	}

	checkCudaErrors(cudaMalloc(&d_solutionV, sizeof(double) * this->n_dofs * dim));
	checkCudaErrors(cudaMalloc(&d_oldSolutionV, sizeof(double) * this->n_dofs * dim));
}

template <int dim>
void cudaPfem2Fem<dim>::initialize_fem_solution()
{
	pfem2Fem<dim>::initialize_fem_solution();

	const Tensor<1, dim> initialVelocity = this->mainSolver->getParameterHandler().getVelocityInitialValue();
	for (int i = 0; i < dim; ++i){
		hostSolutionV[i] = initialVelocity[i];
		hostOldSolutionV[i] = initialVelocity[i];

		checkCudaErrors(cudaMemcpy(d_solutionV + i * this->n_dofs, hostSolutionV[i].data(), sizeof(double) * this->n_dofs, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_oldSolutionV + i * this->n_dofs, hostOldSolutionV[i].data(), sizeof(double) * this->n_dofs, cudaMemcpyHostToDevice));
	}
}

template<int dim>
const cudaPfem2Cell<dim>* cudaPfem2Fem<dim>::getCells() const
{
	return d_cells;
}

template <int dim>
const double *cudaPfem2Fem<dim>::getDeviceSolutionV() const
{
    return d_solutionV;
}

template <int dim>
const double *cudaPfem2Fem<dim>::getDeviceOldSolutionV() const
{
    return d_oldSolutionV;
}
