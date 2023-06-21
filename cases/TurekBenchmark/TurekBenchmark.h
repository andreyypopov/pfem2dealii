#include "../../src/pfem2solver.h"
#include "../../src/pfem2fem.h"

class parabolicBC : public Function<2>
{
public:
	parabolicBC() : Function<2>() {}

	virtual double value (const Point<2> &p, const unsigned int component = 0) const;
};

class TurekBenchmarkFemSolver : public pfem2Fem<2>
{
public:
	TurekBenchmarkFemSolver(const FE_Q<2> *finite_element);
	virtual ~TurekBenchmarkFemSolver();

	virtual void setup_velocity_constraints() override;
	virtual void fill_velocity_boundary_dofs_list() override;
};

class TurekBenchmarkSolver : public pfem2Solver<2>
{
public:
	TurekBenchmarkSolver(pfem2Fem<2> *femSolver,
		pfem2ParticleHandler<2> *particleHandler, pfem2ParameterHandler<2> *parameterHandler);

	virtual const double velocityDirichletBC(unsigned int boundaryID, unsigned int component = 0) const override;
	virtual const double pressureDirichletBC(unsigned int boundaryID) const override;

private:

};
