#include "pfem2parameterhandler.h"

#include <deal.II/base/parameter_handler.h>

template<int dim>
pfem2ParameterHandler<dim>::pfem2ParameterHandler()
    : dynamicViscosity(1e-3)
    , fluidDensity(1.0)
    , timeStep(0.1)
    , finalTime(1.0)
	, outerIterations(2)
    , solverTolerance(1e-7)
    , maxSolverIterations(10000)
	, maxParticlesPerCellPart(2)
	, particleIntegrationSteps(5)
    , resultsOutputFrequency(10)
    , outputParticles(false)
    , boundaryForForcesComputation(0)
{
	for (int i = 0; i < dim; ++i)
		particlesPerCell[i] = 2;
}

template<int dim>
void pfem2ParameterHandler<dim>::readParameters(const std::string& filename)
{
	ParameterHandler prm;

	//1. Declaration of parameters
	//TO DO: Set parameter types patterns (integer/double/bool)
	prm.enter_subsection("Fluid characteristics");
	{
		prm.declare_entry ("Dynamic viscosity", "1.0");
		prm.declare_entry ("Density", "1.0");
	}
	prm.leave_subsection();

	prm.enter_subsection("Computation parameters");
	{
		prm.declare_entry ("Time step", "0.1");
		prm.declare_entry ("Final Time", "1.0");
		prm.declare_entry ("Outer iterations", "2");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Linear solver parameters");
	{
		prm.declare_entry ("Tolerance", "1e-7");
		prm.declare_entry ("Number of iterations", "10000");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Initial conditions");
	{
		prm.declare_entry ("Initial pressure", "0.0");
		prm.declare_entry ("Initial velocity x", "0.0");
		if (dim > 1)
			prm.declare_entry ("Initial velocity y", "0.0");
		if (dim > 2)
			prm.declare_entry ("Initial velocity z", "0.0");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Particles");
	{
		prm.declare_entry ("Number of particles in the x direction", "2");
		if (dim > 1)
			prm.declare_entry ("Number of particles in the y direction", "2");
		if (dim > 2)
			prm.declare_entry ("Number of particles in the z direction", "2");
	}
	prm.declare_entry ("Maximum number of particles per cell part", "2");
	prm.declare_entry ("Particle integration steps", "5");
	prm.leave_subsection();
	
	prm.enter_subsection("Input and output parameters");
	{
		prm.declare_entry ("Mesh file name", "");
		prm.declare_entry ("Results output frequency", "10");
		prm.declare_entry ("Output particles", "false");
		prm.declare_entry ("Boundary id for forces computation", "0");
	}
	prm.leave_subsection();
	
	//2. Parsing input file (undefined entries are allowed to be skipped)
	prm.parse_input (filename, "", true);	
	
	//3. Reading parameters from file
	prm.enter_subsection("Fluid characteristics");
	{
		dynamicViscosity = prm.get_double ("Dynamic viscosity");
		fluidDensity = prm.get_double ("Density");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Computation parameters");
	{
		timeStep = prm.get_double("Time step");
		finalTime = prm.get_double("Final Time");
		outerIterations = prm.get_integer("Outer iterations");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Linear solver parameters");
	{
		solverTolerance = prm.get_double("Tolerance");
		maxSolverIterations = prm.get_integer("Number of iterations");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Initial conditions");
	{
		pressureInitialValue = prm.get_double("Initial pressure");
		velocityInitialValue[0] = prm.get_double("Initial velocity x");
		if (dim > 1)
			velocityInitialValue[1] = prm.get_double("Initial velocity y");
		if (dim > 2)
			velocityInitialValue[2] = prm.get_double("Initial velocity z");
	}
	prm.leave_subsection();
		
	prm.enter_subsection("Particles");
	{
		particlesPerCell[0] = prm.get_integer("Number of particles in the x direction");
		if (dim > 1)
			particlesPerCell[1] = prm.get_integer("Number of particles in the y direction");
		if (dim > 2)
			particlesPerCell[2] = prm.get_integer("Number of particles in the z direction");
	}
	maxParticlesPerCellPart = prm.get_integer ("Maximum number of particles per cell part");
	particleIntegrationSteps = prm.get_integer ("Particle integration steps");
	prm.leave_subsection();
	
	prm.enter_subsection("Input and output parameters");
	{
		meshFileName = prm.get("Mesh file name");
		resultsOutputFrequency = prm.get_integer("Results output frequency");
		outputParticles = prm.get_bool("Output particles");
		boundaryForForcesComputation = prm.get_integer("Boundary id for forces computation");
	}
	prm.leave_subsection();
}

template <int dim>
const double& pfem2ParameterHandler<dim>::getDynamicViscosity() const
{
    return dynamicViscosity;
}

template <int dim>
const double &pfem2ParameterHandler<dim>::getFluidDensity() const
{
    return fluidDensity;
}

template <int dim>
const double &pfem2ParameterHandler<dim>::getTimeStep() const
{
    return timeStep;
}

template <int dim>
const double &pfem2ParameterHandler<dim>::getFinalTime() const
{
    return finalTime;
}

template <int dim>
const unsigned int &pfem2ParameterHandler<dim>::getOuterIterations() const
{
    return outerIterations;
}

template <int dim>
const double &pfem2ParameterHandler<dim>::getSolverTolerance() const
{
    return solverTolerance;
}

template <int dim>
const unsigned int &pfem2ParameterHandler<dim>::getMaxSolverIterations() const
{
    return maxSolverIterations;
}

template <int dim>
const double &pfem2ParameterHandler<dim>::getPressureInitialValue() const
{
    return pressureInitialValue;
}

template <int dim>
const Tensor<1, dim> &pfem2ParameterHandler<dim>::getVelocityInitialValue() const
{
    return velocityInitialValue;
}

template <int dim>
const std::array<unsigned int, dim> & pfem2ParameterHandler<dim>::getParticlesPerCell() const
{
    return particlesPerCell;
}

template <int dim>
const unsigned int &pfem2ParameterHandler<dim>::getMaxParticlesPerCellPart() const
{
    return maxParticlesPerCellPart;
}

template <int dim>
const unsigned int &pfem2ParameterHandler<dim>::getParticleIntegrationSteps() const
{
    return particleIntegrationSteps;
}

template <int dim>
const std::string &pfem2ParameterHandler<dim>::getMeshFileName() const
{
    return meshFileName;
}

template <int dim>
const bool &pfem2ParameterHandler<dim>::getOutputParticles() const
{
    return outputParticles;
}

template <int dim>
const unsigned int &pfem2ParameterHandler<dim>::getBoundaryForForcesComputation() const
{
    return boundaryForForcesComputation;
}

template <int dim>
const unsigned int &pfem2ParameterHandler<dim>::getResultsOutputFrequency() const
{
    return resultsOutputFrequency;
}
