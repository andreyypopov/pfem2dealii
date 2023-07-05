#include "../../src/pfem2solver.h"
#include "../../src/cuda/cudapfem2fem.cuh"

class parabolicBC : public Function<2>
{
public:
	parabolicBC() : Function<2>() {}

	virtual double value (const Point<2> &p, const unsigned int component = 0) const;
};

class TurekBenchmarkFemSolver : public cudaPfem2Fem<2>
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

	virtual double velocityDirichletBC(unsigned int boundaryID, unsigned int component = 0) const override;
	virtual double pressureDirichletBC(unsigned int boundaryID) const override;

private:

};
