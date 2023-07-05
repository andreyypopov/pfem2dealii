#include "TurekBenchmark.cuh"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "../../src/cuda/cudapfem2particlehandler.cuh"

double parabolicBC::value(const Point<2> &p, const unsigned int) const
{
	return 4 * 1.5 * p[1] * (0.41 - p[1]) / (0.41 * 0.41);
}

TurekBenchmarkSolver::TurekBenchmarkSolver(pfem2Fem<2> *femSolver,
	pfem2ParticleHandler<2> *particleHandler, pfem2ParameterHandler<2> *parameterHandler)
	: pfem2Solver(femSolver, particleHandler, parameterHandler)
{
	velocityDirichletBCpatchIDs = { 1, 3, 4 };
	pressureDirichletBCpatchIDs = { 2 };
}

double TurekBenchmarkSolver::velocityDirichletBC(unsigned int boundaryID, unsigned int component) const
{
	switch (boundaryID)
	{
	case 1:
		return (component == 0) ? 1.0 : 0.0;
	case 3:
	case 4:
		return 0;
	default:
		return 0;
	}
}

double TurekBenchmarkSolver::pressureDirichletBC(unsigned int boundaryID) const
{
	return 0;
}

TurekBenchmarkFemSolver::TurekBenchmarkFemSolver(const FE_Q<2> *finite_element)
	: cudaPfem2Fem<2>(finite_element)
{
}

TurekBenchmarkFemSolver::~TurekBenchmarkFemSolver()
{
}

void TurekBenchmarkFemSolver::setup_velocity_constraints()
{
	for(int i = 0; i < 2; ++i){
		constraintsPredV[i].clear ();
    	constraintsPredV[i].reinit (locally_relevant_dofs);
    	DoFTools::make_hanging_node_constraints(dof_handler, constraintsPredV[i]);
		for(unsigned int boundaryID : mainSolver->getVelocityDirichletBCpatchIDs()){
			if(i == 0 && boundaryID == 1)
				VectorTools::interpolate_boundary_values (dof_handler, boundaryID, parabolicBC(), constraintsPredV[i]);
			else
				VectorTools::interpolate_boundary_values (dof_handler, boundaryID, ConstantFunction<2>(mainSolver->velocityDirichletBC(boundaryID, i)), constraintsPredV[i]);
		}
		constraintsPredV[i].close ();

		constraintsV[i].clear ();
    	constraintsV[i].reinit (locally_relevant_dofs);
    	DoFTools::make_hanging_node_constraints(dof_handler, constraintsV[i]);
		for(unsigned int boundaryID : mainSolver->getVelocityDirichletBCpatchIDs()){
			if(i == 0 && boundaryID == 1)
				VectorTools::interpolate_boundary_values (dof_handler, boundaryID, parabolicBC(), constraintsV[i]);
			else
				VectorTools::interpolate_boundary_values (dof_handler, boundaryID, ConstantFunction<2>(mainSolver->velocityDirichletBC(boundaryID, i)), constraintsV[i]);
		}
		constraintsV[i].close ();
	}
}

void TurekBenchmarkFemSolver::fill_velocity_boundary_dofs_list()
{
	parabolicBC inflowBoundaryCondition;

	for(const auto &cell : dof_handler.active_cell_iterators())
		if(cell->is_locally_owned())
			for (unsigned int face_number = 0; face_number < GeometryInfo<2>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary() && mainSolver->getVelocityDirichletBCpatchIDs().count(cell->face(face_number)->boundary_id()))
					for(int vert = 0; vert < GeometryInfo<2>::vertices_per_face; ++vert){
						Tensor<1, 2> velocityVector;

						for(int i = 0; i < 2; ++i){
							if(i == 0 && cell->face(face_number)->boundary_id() == 1)
								velocityVector[i] = inflowBoundaryCondition.value(cell->face(face_number)->vertex(vert));
							else
								velocityVector[i] = mainSolver->velocityDirichletBC(cell->face(face_number)->boundary_id(), i);
						}

						velocityDirichletBoundaryDoFs[cell->face(face_number)->vertex_dof_index(vert,0)] = velocityVector;
					}
}

int main (int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

	FE_Q<2> finiteElement(1);
	TurekBenchmarkFemSolver fem(&finiteElement);
	cudaPfem2ParticleHandler<2> particleHandler(&finiteElement);
	pfem2ParameterHandler<2> parameterHandler;

	TurekBenchmarkSolver TurekBenchmarkProblem(&fem, &particleHandler, &parameterHandler);
	TurekBenchmarkProblem.run();

	return 0;
}
