#include "../../src/pfem2solver.h"

class TurekBenchmarkSolver : public pfem2Solver<2>
{
public:
	TurekBenchmarkSolver(pfem2Fem<2> *femSolver,
		pfem2ParticleHandler<2> *particleHandler, pfem2ParameterHandler<2> *parameterHandler);

	virtual const double velocityDirichletBC(unsigned int boundaryID, unsigned int component = 0) const override;
	virtual const double pressureDirichletBC(unsigned int boundaryID) const override;

private:

};
