#include "TurekBenchmark.h"

TurekBenchmarkSolver::TurekBenchmarkSolver(pfem2Fem<2> *femSolver,
	pfem2ParticleHandler<2> *particleHandler, pfem2ParameterHandler<2> *parameterHandler)
	: pfem2Solver(femSolver, particleHandler, parameterHandler)
{
	velocityDirichletBCpatchIDs = { 1, 3, 4 };
	pressureDirichletBCpatchIDs = { 2 };
}

const double TurekBenchmarkSolver::velocityDirichletBC(unsigned int boundaryID, unsigned int component) const
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

const double TurekBenchmarkSolver::pressureDirichletBC(unsigned int boundaryID) const
{
	return 0;
}

int main (int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

	FE_Q<2> finiteElement(1);
	pfem2Fem<2> fem(&finiteElement);
	pfem2ParticleHandler<2> particleHandler(&finiteElement);
	pfem2ParameterHandler<2> parameterHandler;

	TurekBenchmarkSolver TurekBenchmarkProblem(&fem, &particleHandler, &parameterHandler);
	TurekBenchmarkProblem.run();

	return 0;
}
