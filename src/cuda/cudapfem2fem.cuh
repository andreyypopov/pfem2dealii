#ifndef CUDAPFEM2FEM_H
#define CUDAPFEM2FEM_H

#include "../pfem2fem.h"

using namespace dealii;

template<int dim>
class cudaPfem2Cell
{
public:
	cudaPfem2Cell(const int index);

	void set_vertex_index(const unsigned int vertex_index, int i);
	void set_dof_index(const unsigned int dof_index, int i);
	void set_vertex_coords(const Point<dim> &p, int i);

	__host__ __device__ const double* get_vertex_coords() const;

private:
	int index;
	unsigned int vertex_indices[GeometryInfo<dim>::vertices_per_cell];
	double vertex_coords[GeometryInfo<dim>::vertices_per_cell * dim];
	types::global_dof_index dof_indices[GeometryInfo<dim>::vertices_per_cell];
};

template<int dim>
class cudaPfem2Fem : public pfem2Fem<dim>
{
public:
	cudaPfem2Fem(const FE_Q<dim> *finite_element);
	virtual ~cudaPfem2Fem();

	virtual void setup_system();

	const cudaPfem2Cell<dim>* getCells() const;

private:
	Point<dim> *d_vertices;
	cudaPfem2Cell<dim> *d_cells;

	double *d_solutionV[dim];
	double *d_oldSolutionV[dim];
};

template class cudaPfem2Cell<2>;
template class cudaPfem2Cell<3>;

template class cudaPfem2Fem<2>;
template class cudaPfem2Fem<3>;

#endif // CUDAPFEM2PARTICLEHANDLER_H
