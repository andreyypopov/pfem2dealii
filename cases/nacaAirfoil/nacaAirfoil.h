#include "../../src/pfem2parameterhandler.h"
#include "../../src/pfem2solver.h"

class nacaAirfoilParameterHandler : public pfem2ParameterHandler<2>
{
public:
    virtual void readParameters(const std::string& filename);

    const double& getFreeStreamVelocity() const;
    const double& getAngleOfAttack() const;

private:
    //free-stream flow parameters
    double freeStreamVelocity;
    double angleOfAttack;
};

class nacaAirfoilSolver : public pfem2Solver<2>
{
public:
	nacaAirfoilSolver(pfem2Fem<2> *femSolver,
		pfem2ParticleHandler<2> *particleHandler, pfem2ParameterHandler<2> *parameterHandler);

	virtual const double velocityDirichletBC(unsigned int boundaryID, unsigned int component = 0) const override;
	virtual const double pressureDirichletBC(unsigned int boundaryID) const override;
	
	void setFreeStreamParameters();
	
private:
    double freeStreamVelocity;
    double angleOfAttack;
};
