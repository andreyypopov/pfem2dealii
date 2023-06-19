#ifndef PFEM2FEM_H
#define PFEM2FEM_H

#include <array>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

using namespace dealii;

const int QUAD_POINTS_PER_DIRECTION = 2;

template<int dim>
class pfem2Solver;

template<int dim>
class pfem2Fem
{
public:
	pfem2Fem(const FE_Q<dim> *finite_element);
	virtual ~pfem2Fem();

	virtual void setup_system();
	virtual void initialize_fem_solution();
	virtual void fem_step();
	virtual void output_fem_solution(int timestep_number, bool exportPrediction = false);

	void setPfem2Solver(pfem2Solver<dim>* mainSolver);

	const DoFHandler<dim> &getDoFhandler() const;

	const std::array<TrilinosWrappers::MPI::Vector, dim>& getSolutionV() const;
	const std::array<TrilinosWrappers::MPI::Vector, dim>& getOldSolutionV() const;
	const TrilinosWrappers::MPI::Vector& getSolutionP() const;

	void setSolutionV(const TrilinosWrappers::MPI::Vector& solutionV, unsigned int component = 0);
	void setSolutionP(const TrilinosWrappers::MPI::Vector& solutionP);

	const unsigned int &getDofsPerCell() const;
	const IndexSet &getLocallyOwnedDofs() const;

private:
	virtual void velocity_prediction_bc();
	virtual void assemble_velocity_prediction();
	virtual void assemble_pressure_equation();
	virtual void assemble_velocity_correction();

	void solve_velocity(bool correction = false);
	void solve_pressure();
	
	DoFHandler<dim> dof_handler;

	QGauss<dim> quadrature_formula;
	QGauss<dim-1> face_quadrature_formula;

	const FE_Q<dim> *feq;

	FEValues<dim> fe_values;
	FEFaceValues<dim> fe_face_values;

	const unsigned int n_q_points, n_face_q_points;

	std::array<FullMatrix<double>, dim> local_matrixV;
	FullMatrix<double> local_matrixP;

	std::array<Vector<double>, dim> local_rhsV;
	Vector<double> local_rhsP;

	std::array<TrilinosWrappers::SparseMatrix, dim> system_mPredV;
	std::array<TrilinosWrappers::SparseMatrix, dim> system_mV;
	TrilinosWrappers::SparseMatrix system_mP;
	std::array<TrilinosWrappers::MPI::Vector, dim> system_rV;
	TrilinosWrappers::MPI::Vector system_rP;

	std::array<TrilinosWrappers::MPI::Vector, dim> locally_relevant_solutionV;
	TrilinosWrappers::MPI::Vector locally_relevant_solutionP;
	TrilinosWrappers::MPI::Vector locally_relevant_old_solutionP;
	std::array<TrilinosWrappers::MPI::Vector, dim> locally_relevant_predictionV;
	std::array<TrilinosWrappers::MPI::Vector, dim> locally_relevant_old_solutionV;

	TrilinosWrappers::MPI::Vector velocityBcWeights;
	std::map<unsigned int, Tensor<1,dim>> velocityDirichletBoundaryDoFs;

	const unsigned int dofs_per_cell;
	std::vector<types::global_dof_index> local_dof_indices;

	IndexSet locally_owned_dofs;
	IndexSet locally_relevant_dofs;

	std::array<AffineConstraints<double>, dim> constraintsPredV;
	AffineConstraints<double> constraintsP;
	std::array<AffineConstraints<double>, dim> constraintsV;
	
	SolverControl solver_control;
	TrilinosWrappers::SolverBase *trilinosSolver;
	TrilinosWrappers::PreconditionBase *preconditionerV;
	TrilinosWrappers::PreconditionBase *preconditionerP;
	
	double mu;
	double rho;
	unsigned int outerCorrections;
	
	pfem2Solver<dim>* mainSolver;
};

template class pfem2Fem<2>;

#endif // PFEM2FEM_H
