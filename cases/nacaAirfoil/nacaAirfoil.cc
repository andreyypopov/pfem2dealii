#include "nacaAirfoil.h"

#include <deal.II/base/parameter_handler.h>

void nacaAirfoilParameterHandler::readParameters(const std::string &filename)
{
    pfem2ParameterHandler::readParameters(filename);

    ParameterHandler prm;
    prm.enter_subsection("Free-stream flow");
	{
		prm.declare_entry ("Velocity", "10.0");
		prm.declare_entry ("Angle of attack", "4.0");
	}
	prm.leave_subsection();

	prm.parse_input (filename, "", true);	

    prm.enter_subsection("Free-stream flow");
	{
		freeStreamVelocity = prm.get_double ("Velocity");
		angleOfAttack = prm.get_double ("Angle of attack") / 180.0 * M_PI;
	}
	prm.leave_subsection();
}

const double &nacaAirfoilParameterHandler::getFreeStreamVelocity() const
{
    return freeStreamVelocity;
}

const double &nacaAirfoilParameterHandler::getAngleOfAttack() const
{
    return angleOfAttack;
}

nacaAirfoilSolver::nacaAirfoilSolver(pfem2Fem<2> *femSolver,
	pfem2ParticleHandler<2> *particleHandler, pfem2ParameterHandler<2> *parameterHandler)
	: pfem2Solver(femSolver, particleHandler, parameterHandler)
{
	velocityDirichletBCpatchIDs = { 1, 3, 6 };
	pressureDirichletBCpatchIDs = { 2 };
}

const double nacaAirfoilSolver::velocityDirichletBC(unsigned int boundaryID, unsigned int component) const
{
	switch (boundaryID)
	{
	case 1:
	case 6:
		return freeStreamVelocity * ((component == 0) ? cos(angleOfAttack) : sin(angleOfAttack));
	case 3:
		return 0;
	default:
		return 0;
	}
}

const double nacaAirfoilSolver::pressureDirichletBC(unsigned int boundaryID) const
{
	return 0;
}

void nacaAirfoilSolver::setFreeStreamParameters()
{
	this->freeStreamVelocity = static_cast<nacaAirfoilParameterHandler*>(parameterHandler)->getFreeStreamVelocity();
	this->angleOfAttack = static_cast<nacaAirfoilParameterHandler*>(parameterHandler)->getAngleOfAttack();
}

int main (int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
	
	FE_Q<2> finiteElement(1);
	pfem2Fem<2> fem(&finiteElement);
	pfem2ParticleHandler<2> particleHandler(&finiteElement);
	
	nacaAirfoilParameterHandler parameterHandler;
	nacaAirfoilSolver nacaAirfoilProblem(&fem, &particleHandler, &parameterHandler);
	nacaAirfoilProblem.setFreeStreamParameters();
	nacaAirfoilProblem.run();
  
	return 0;
}
