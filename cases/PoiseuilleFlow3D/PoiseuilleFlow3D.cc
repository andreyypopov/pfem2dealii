#include "PoiseuilleFlow3D.h"

#include <deal.II/grid/grid_generator.h>

PoiseuilleFlow3DSolver::PoiseuilleFlow3DSolver(pfem2Fem<3> *femSolver,
	pfem2ParticleHandler<3> *particleHandler, pfem2ParameterHandler<3> *parameterHandler)
	: pfem2Solver(femSolver, particleHandler, parameterHandler)
{
	velocityDirichletBCpatchIDs = { 2, 3, 4, 5 };
	pressureDirichletBCpatchIDs = { 0, 1 };
}

double PoiseuilleFlow3DSolver::velocityDirichletBC(unsigned int boundaryID, unsigned int component) const
{
	return 0;
}

double PoiseuilleFlow3DSolver::pressureDirichletBC(unsigned int boundaryID) const
{
	return (boundaryID == 0) ? 100.0 : 0.0;
}

void PoiseuilleFlow3DSolver::build_mesh(bool outputAfterBuild)
{
	TimerOutput::Scope timer_section(timer, "Mesh construction");

    const Point<3> bottom_left(0, -1, -1);
    const Point<3> top_right(10, 1, 1);
    std::vector<unsigned int> repetitions {100, 20, 20};
    GridGenerator::subdivided_hyper_rectangle(tria, repetitions, bottom_left, top_right, true);

    pcout << "Grid contains " << tria.n_cells() << " cells" << std::endl;
}

int main (int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

	FE_Q<3> finiteElement(1);
	pfem2Fem<3> fem(&finiteElement);
	pfem2ParticleHandler<3> particleHandler(&finiteElement);
	pfem2ParameterHandler<3> parameterHandler;

	PoiseuilleFlow3DSolver PoiseuilleFlow3DProblem(&fem, &particleHandler, &parameterHandler);
	PoiseuilleFlow3DProblem.run();

	return 0;
}
