#ifndef PFEM2PARAMETERHANDLER_H
#define PFEM2PARAMETERHANDLER_H

#include <string>
#include <array>

#include <deal.II/base/tensor.h>

using namespace dealii;

template<int dim>
class pfem2ParameterHandler
{
public:
    pfem2ParameterHandler();
    
    virtual void readParameters(const std::string& filename);

    const double& getDynamicViscosity() const;
    const double& getFluidDensity() const;
    const double& getTimeStep() const;
    const double& getFinalTime() const;
    const unsigned int& getOuterIterations() const;
    const double& getSolverTolerance() const;
    const unsigned int& getMaxSolverIterations() const;
    const double& getPressureInitialValue() const;
    const Tensor<1,dim>& getVelocityInitialValue() const;
    const std::array<unsigned int, dim>& getParticlesPerCell() const;
    const unsigned int& getMaxParticlesPerCellPart() const;
    const unsigned int& getParticleIntegrationSteps() const;
    const std::string& getMeshFileName() const;
    const unsigned int& getResultsOutputFrequency() const;
    const bool& getOutputParticles() const;
    const unsigned int& getLoadsBoundaryID() const;
    const double& getThickness() const;
    const double& getMeanVelocity() const;

private:
    //fluid properties
    double dynamicViscosity;
    double fluidDensity;

    //computation parameters
    double timeStep;
    double finalTime;
    unsigned int outerIterations;

    //linear solver parameters
    double solverTolerance;
    unsigned int maxSolverIterations;

    //initial conditions
    double pressureInitialValue;
    Tensor<1,dim> velocityInitialValue;

    //particle distribution parameters
    std::array<unsigned int, dim> particlesPerCell;
    unsigned int maxParticlesPerCellPart;
    unsigned int particleIntegrationSteps;

    //input/output parameters
    std::string meshFileName;
    unsigned int resultsOutputFrequency;
    bool outputParticles;

    //loads calculation
    unsigned int loadsBoundaryID;
    double thickness;
    double meanVelocity;
};

template class pfem2ParameterHandler<2>;

template class pfem2ParameterHandler<3>;

#endif // PFEM2PARAMETERHANDLER_H
