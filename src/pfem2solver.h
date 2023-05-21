#ifndef PFEM2SOLVER_H
#define PFEM2SOLVER_H

#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include "pfem2fem.h"
#include "pfem2particlehandler.h"
#include "pfem2parameterhandler.h"

using namespace dealii;

template<int dim> class pfem2ParticleHandler;

template<int dim>
class pfem2Solver
{
public:
	pfem2Solver(const pfem2Fem<dim> *femSolver, const pfem2ParticleHandler<dim> *particleHandler);
	virtual ~pfem2Solver();
	
	const MPI_Comm getCommunicator() const;
	const ConditionalOStream& getPcout() const;
	const TimerOutput* getTimer() const;

	const parallel::distributed::Triangulation<dim>& getTriangulation() const;

	const pfem2Fem<dim> *getFemSolver() const;
	const pfem2ParameterHandler<dim> &getParameterHandler() const;

	const std::set<unsigned int> &getVelocityDirichletBCpatchIDs() const;
	const std::set<unsigned int> &getPressureDirichletBCpatchIDs() const;

	const int& getTimestepNumber() const;

	virtual const double velocityDirichletBC(unsigned int boundaryID, unsigned int component = 0) = 0;
	virtual const double pressureDirichletBC(unsigned int boundaryID) = 0;

private:
	virtual void build_mesh (const std::string &filename, bool outputAfterBuild = false);
	virtual void run();
	virtual void output_results(bool exportParticles, bool exportPrediction = false);
	
	MPI_Comm mpi_communicator;
	const unsigned int n_mpi_processes;
	const unsigned int this_mpi_process;
	ConditionalOStream pcout;
	TimerOutput *timer;
	
	parallel::distributed::Triangulation<dim> tria;
		
	pfem2Fem<dim> *femSolver;
	pfem2ParticleHandler<dim> *particleHandler;
	pfem2ParameterHandler<dim> parameterHandler;

	std::set<unsigned int> velocityDirichletBCpatchIDs;
	std::set<unsigned int> pressureDirichletBCpatchIDs;

	double time, time_step, final_time;
	int timestep_number;
};

#endif // PFEM2SOLVER_H
