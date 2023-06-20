#include "../../src/pfem2solver.h"

class PoiseuilleFlow3DSolver : public pfem2Solver<3>
{
public:
	PoiseuilleFlow3DSolver(pfem2Fem<3> *femSolver,
		pfem2ParticleHandler<3> *particleHandler, pfem2ParameterHandler<3> *parameterHandler);

	virtual const double velocityDirichletBC(unsigned int boundaryID, unsigned int component = 0) const override;
	virtual const double pressureDirichletBC(unsigned int boundaryID) const override;

private:
	virtual void build_mesh (bool outputAfterBuild = false);
};
