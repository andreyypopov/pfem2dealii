#include "pfem2fem.h"

#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include "pfem2parameterhandler.h"
#include "pfem2solver.h"

template<int dim>
pfem2Fem<dim>::pfem2Fem(const FE_Q<dim> *finite_element)
	: quadrature_formula(QUAD_POINTS_PER_DIRECTION)
	, face_quadrature_formula(QUAD_POINTS_PER_DIRECTION)
	, feq(finite_element)
	, fe_values (*feq, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values)
	, fe_face_values (*feq, face_quadrature_formula, update_values | update_gradients | update_quadrature_points | update_normal_vectors | update_JxW_values)
    , n_q_points (quadrature_formula.size())
	, n_face_q_points (face_quadrature_formula.size())
	, dofs_per_cell (feq->dofs_per_cell)
	, local_dof_indices (dofs_per_cell)
	, solver_control (10000, 1e-7)
{
	for (int i = 0; i < dim; ++i){
		local_matrixV[i] = FullMatrix<double>(dofs_per_cell);
		local_rhsV[i] = Vector<double>(dofs_per_cell);
	}

	local_matrixP = FullMatrix<double>(dofs_per_cell);
	local_rhsP = Vector<double>(dofs_per_cell);

	this->trilinosSolver = new TrilinosWrappers::SolverGMRES(solver_control);
	this->preconditionerV = new TrilinosWrappers::PreconditionJacobi;
	this->preconditionerP = new TrilinosWrappers::PreconditionAMG;
	
	this->mainSolver = nullptr;
}

template<int dim>
pfem2Fem<dim>::~pfem2Fem()
{
	delete trilinosSolver;	
	delete preconditionerV;
	delete preconditionerP;
}

template<int dim>
void pfem2Fem<dim>::setup_system()
{
	TimerOutput::Scope timer_section(mainSolver->getTimer(), "System setup");

	dof_handler.distribute_dofs(*feq);
	mainSolver->getPcout() << "Number of degrees of freedom for each field: " << dof_handler.n_dofs() << std::endl;

	locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (dof_handler, locally_relevant_dofs);

    //setup constraints for boundary conditions for velocity field
	setup_velocity_constraints();

	//setup constraints for boundary conditions for pressure field
	setup_pressure_constraints();

	//setup matrices and vectors for velocity field
	for(int i = 0; i < dim; ++i){
		locally_relevant_solutionV[i].reinit(locally_owned_dofs, locally_relevant_dofs, mainSolver->getCommunicator());
		locally_relevant_old_solutionV[i].reinit(locally_owned_dofs, locally_relevant_dofs, mainSolver->getCommunicator());
		locally_relevant_predictionV[i].reinit(locally_owned_dofs, locally_relevant_dofs, mainSolver->getCommunicator());
		system_rV[i].reinit(locally_owned_dofs, mainSolver->getCommunicator());

		DynamicSparsityPattern dspV(locally_relevant_dofs);

    	DoFTools::make_sparsity_pattern (dof_handler, dspV, constraintsPredV[i], false);
    	SparsityTools::distribute_sparsity_pattern (dspV, locally_owned_dofs, mainSolver->getCommunicator(), locally_relevant_dofs);
    	system_mPredV[i].reinit (locally_owned_dofs, locally_owned_dofs, dspV, mainSolver->getCommunicator());

		DoFTools::make_sparsity_pattern (dof_handler, dspV, constraintsV[i], false);
    	SparsityTools::distribute_sparsity_pattern (dspV, locally_owned_dofs, mainSolver->getCommunicator(), locally_relevant_dofs);
    	system_mV[i].reinit (locally_owned_dofs, locally_owned_dofs, dspV, mainSolver->getCommunicator());
	}

	//setup matrix and vectors for pressure field
	locally_relevant_solutionP.reinit(locally_owned_dofs, locally_relevant_dofs, mainSolver->getCommunicator());
	locally_relevant_old_solutionP.reinit(locally_owned_dofs, locally_relevant_dofs, mainSolver->getCommunicator());
	system_rP.reinit(locally_owned_dofs, mainSolver->getCommunicator());

	DynamicSparsityPattern dspP(locally_relevant_dofs);

	DoFTools::make_sparsity_pattern (dof_handler, dspP, constraintsP, false);
	SparsityTools::distribute_sparsity_pattern (dspP, locally_owned_dofs, mainSolver->getCommunicator(), locally_relevant_dofs);
	system_mP.reinit (locally_owned_dofs, locally_owned_dofs, dspP, mainSolver->getCommunicator());

	//fill the list of boundary DoFs where Dirichlet conditions are set for the velocity field (number of DoF + velocity BC vector)
	fill_velocity_boundary_dofs_list();
}

template<int dim>
void pfem2Fem<dim>::setup_velocity_constraints()
{
	for(int i = 0; i < dim; ++i){
		constraintsPredV[i].clear ();
    	constraintsPredV[i].reinit (locally_relevant_dofs);
    	DoFTools::make_hanging_node_constraints(dof_handler, constraintsPredV[i]);
		for(unsigned int boundaryID : mainSolver->getVelocityDirichletBCpatchIDs())
			VectorTools::interpolate_boundary_values (dof_handler, boundaryID, ConstantFunction<dim>(mainSolver->velocityDirichletBC(boundaryID, i)), constraintsPredV[i]);
		constraintsPredV[i].close ();

		constraintsV[i].clear ();
    	constraintsV[i].reinit (locally_relevant_dofs);
    	DoFTools::make_hanging_node_constraints(dof_handler, constraintsV[i]);
		for(unsigned int boundaryID : mainSolver->getVelocityDirichletBCpatchIDs())
			VectorTools::interpolate_boundary_values (dof_handler, boundaryID, ConstantFunction<dim>(mainSolver->velocityDirichletBC(boundaryID, i)), constraintsV[i]);
		constraintsV[i].close ();
	}
}

template<int dim>
void pfem2Fem<dim>::setup_pressure_constraints()
{
	constraintsP.clear ();
    constraintsP.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraintsP);
	for(unsigned int boundaryID : mainSolver->getPressureDirichletBCpatchIDs())
		VectorTools::interpolate_boundary_values (dof_handler, boundaryID, ConstantFunction<dim>(mainSolver->pressureDirichletBC(boundaryID)), constraintsP);
	constraintsP.close ();
}

template<int dim>
void pfem2Fem<dim>::fill_velocity_boundary_dofs_list()
{
	for(const auto &cell : dof_handler.active_cell_iterators())
		if(cell->is_locally_owned())
			for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary() && mainSolver->getVelocityDirichletBCpatchIDs().count(cell->face(face_number)->boundary_id())){
					Tensor<1, dim> velocityVector;
					for(int i = 0; i < dim; ++i)
						velocityVector[i] = mainSolver->velocityDirichletBC(cell->face(face_number)->boundary_id(), i);
					
					for(int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
						velocityDirichletBoundaryDoFs[cell->face(face_number)->vertex_dof_index(i,0)] = velocityVector;
				}
}

template <int dim>
void pfem2Fem<dim>::velocity_prediction_bc()
{
	const int timestep_number = mainSolver->getTimestepNumber();
	
	TrilinosWrappers::MPI::Vector pressureField = locally_relevant_solutionP;
#ifdef SCHEMEB
	pressureField -= locally_relevant_old_solutionP;
#endif
	
	std::array<TrilinosWrappers::MPI::Vector, dim> pressureGradient;
	std::array<Vector<double>, dim> localPressureGradient;
	Vector<double> localWeights(dofs_per_cell);

	for(int i = 0; i < dim; ++i){
		pressureGradient[i].reinit (locally_relevant_solutionP);
		pressureGradient[i] = 0;
		localPressureGradient[i].reinit (dofs_per_cell);
		localPressureGradient[i] = 0;
	}

	if(timestep_number == 1){
		velocityBcWeights.reinit (locally_relevant_solutionP);
		velocityBcWeights = 0;
	}

	Tensor<1, dim> qPointPressureGradient;
	double shapeValue;
	
    for(const auto &cell : dof_handler.active_cell_iterators())
		if(cell->is_locally_owned()){
			for(int i = 0; i < dim; ++i)
				localPressureGradient[i] = 0.0;
			
			if(timestep_number == 1)
				localWeights = 0.0;
								
			for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
				if(cell->face(face_number)->at_boundary() && mainSolver->getVelocityDirichletBCpatchIDs().count(cell->face(face_number)->boundary_id())){
					fe_face_values.reinit (cell, face_number);
					
					for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point){
						qPointPressureGradient = 0.0;
						
						for (unsigned int i = 0; i < dofs_per_cell; ++i){
							Tensor<1,dim> Ni_p_grad = fe_face_values.shape_grad(i, q_point);
							Ni_p_grad *= pressureField(cell->vertex_dof_index(i,0));
							
							qPointPressureGradient += Ni_p_grad;
						}

						for (unsigned int vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex){
							shapeValue = fe_face_values.shape_value(vertex, q_point);
							
							for(int i = 0; i < dim; ++i)
								localPressureGradient[i](vertex) += shapeValue * qPointPressureGradient[i];
					
							if(timestep_number == 1)
								localWeights(vertex) += shapeValue;
						}
					}
				}
					
			cell->get_dof_indices (local_dof_indices);
			for(int i = 0; i < dim; ++i)
				cell->distribute_local_to_global(localPressureGradient[i], pressureGradient[i]);
			
			if(timestep_number == 1)
				cell->distribute_local_to_global(localWeights, velocityBcWeights);
		}
    
    for(int i = 0; i < dim; ++i)
		pressureGradient[i].compress (VectorOperation::add);

	if(timestep_number == 1)
		velocityBcWeights.compress (VectorOperation::add);

	const double coeff = mainSolver->getParameterHandler().getTimeStep() / rho;
	for(unsigned int i = pressureGradient[0].local_range().first; i < pressureGradient[0].local_range().second; ++i)
		if(velocityDirichletBoundaryDoFs.count(i))
			for(int k = 0; k < dim; ++k)
				pressureGradient[k](i) = pressureGradient[k](i) * coeff / velocityBcWeights(i) + velocityDirichletBoundaryDoFs[i][k];

	for(int i = 0; i < dim; ++i)
    	constraintsPredV[i].clear();
    
    for(unsigned int i = 0; i < pressureGradient[0].size(); ++i)
		if(velocityDirichletBoundaryDoFs.count(i))
			for(int k = 0; k < dim; ++k){
				constraintsPredV[k].add_line(i);
				constraintsPredV[k].set_inhomogeneity(i, pressureGradient[k](i));
			}
    
    for(int i = 0; i < dim; ++i)
		constraintsPredV[i].close();
}

template <int dim>
void pfem2Fem<dim>::assemble_velocity_prediction()
{
	for(int i = 0; i < dim; ++i){
		system_mPredV[i] = 0;
		system_rV[i] = 0;
	}

	double weight, aux;
	unsigned int iDoFindex, jDoFindex;

	const double time_step = mainSolver->getParameterHandler().getTimeStep();
	
	for(const auto &cell : dof_handler.active_cell_iterators())
		if(cell->is_locally_owned()){
			fe_values.reinit (cell);
			for(int i = 0; i < dim; ++i){
				local_matrixV[i] = 0.0;
				local_rhsV[i] = 0.0;
			}

			for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
				weight = fe_values.JxW (q_index);
				
				for (unsigned int i = 0; i < dofs_per_cell; ++i) {
					const Tensor<0,dim> Ni_vel = fe_values.shape_value (i, q_index);
					const Tensor<1,dim> Ni_vel_grad = fe_values.shape_grad (i, q_index);
					
					for (unsigned int j = 0; j < dofs_per_cell; ++j) {
						jDoFindex = cell->vertex_dof_index(j,0);
						
						const Tensor<0,dim> Nj_vel = fe_values.shape_value (j, q_index);
						const Tensor<1,dim> Nj_vel_grad = fe_values.shape_grad (j, q_index);
#ifdef SCHEMEB
						const Tensor<1,dim> Nj_p_grad = fe_values.shape_grad (j, q_index);
#endif
						aux = rho * Ni_vel * Nj_vel * weight;
						for(int k = 0; k < dim; ++k){
							local_matrixV[k](i,j) += aux;
							local_rhsV[k](i) += aux * locally_relevant_old_solutionV[k](jDoFindex);
						}
						
						aux = mu * time_step * weight;
						if(dim == 2){
							//implicit account for tau_ij
							local_matrixV[0](i,j) += aux * (Ni_vel_grad[1] * Nj_vel_grad[1] + 4.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[0]);
							local_matrixV[1](i,j) += aux * (Ni_vel_grad[0] * Nj_vel_grad[0] + 4.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[1]);
													
							//explicit account for tau_ij
							local_rhsV[0](i) -= aux * (Ni_vel_grad[1] * Nj_vel_grad[0] - 2.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[1]) * locally_relevant_solutionV[1](jDoFindex);
							local_rhsV[1](i) -= aux * (Ni_vel_grad[0] * Nj_vel_grad[1] - 2.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[0]) * locally_relevant_solutionV[0](jDoFindex);
						} else if(dim == 3){
							//implicit account for tau_ij
							local_matrixV[0](i,j) += aux * (4.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[0] + Ni_vel_grad[1] * Nj_vel_grad[1] + Ni_vel_grad[2] * Nj_vel_grad[2]);
							local_matrixV[1](i,j) += aux * (Nj_vel_grad[0] * Ni_vel_grad[0] + 4.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[1] + Ni_vel_grad[2] * Nj_vel_grad[2]);
							local_matrixV[2](i,j) += aux * (Nj_vel_grad[0] * Ni_vel_grad[0] + Nj_vel_grad[1] * Ni_vel_grad[1] + 4.0/3.0 * Ni_vel_grad[2] * Nj_vel_grad[2]);

							//explicit account for tau_ij
							local_rhsV[0](i) -= aux * ((Ni_vel_grad[1] * Nj_vel_grad[0] - 2.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[1]) * locally_relevant_solutionV[1](jDoFindex) +
										(Ni_vel_grad[2] * Nj_vel_grad[0] - 2.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[2]) * locally_relevant_solutionV[2](jDoFindex));
							local_rhsV[1](i) -= aux * ((Ni_vel_grad[0] * Nj_vel_grad[1] - 2.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[0]) * locally_relevant_solutionV[0](jDoFindex) +
										(Ni_vel_grad[2] * Nj_vel_grad[1] - 2.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[2]) * locally_relevant_solutionV[2](jDoFindex));
							local_rhsV[2](i) -= aux * ((Ni_vel_grad[0] * Nj_vel_grad[2] - 2.0/3.0 * Ni_vel_grad[2] * Nj_vel_grad[0]) * locally_relevant_solutionV[0](jDoFindex) +
										(Ni_vel_grad[1] * Nj_vel_grad[2] - 2.0/3.0 * Ni_vel_grad[2] * Nj_vel_grad[1]) * locally_relevant_solutionV[1](jDoFindex));
						}
#ifdef SCHEMEB
						aux = time_step * Ni_vel * locally_relevant_old_solutionP(jDoFindex) * weight;
						for(int k = 0; k < dim; ++k)
							local_rhsV[k](i) -= aux * Nj_p_grad[k];
#endif
					}//j
				}//i
			}//q_index

			for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary() && mainSolver->getVelocityDirichletBCpatchIDs().count(cell->face(face_number)->boundary_id()) == 0){
					fe_face_values.reinit (cell, face_number);
					
					if(dim == 2){
						for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point){
							for (unsigned int i = 0; i < dofs_per_cell; ++i)
								for (unsigned int j = 0; j < dofs_per_cell; ++j){
									local_matrixV[0](i,j) -= mu * time_step * fe_face_values.shape_value(i,q_point) *
										4.0 / 3.0 * fe_face_values.shape_grad(j,q_point)[0] * fe_face_values.normal_vector(q_point)[0] * fe_face_values.JxW(q_point);
									local_rhsV[0](i) += mu * time_step * fe_face_values.shape_value(i,q_point) *
										(-2.0 / 3.0) * fe_face_values.shape_grad(j,q_point)[1] * locally_relevant_solutionV[1](cell->vertex_dof_index(j,0)) *
											fe_face_values.normal_vector(q_point)[0] * fe_face_values.JxW(q_point);
											
									local_matrixV[1](i,j) -= mu * time_step * fe_face_values.shape_value(i,q_point) *
										fe_face_values.shape_grad(j,q_point)[0] * fe_face_values.normal_vector(q_point)[0] * fe_face_values.JxW(q_point);
									local_rhsV[1](i) += mu * time_step * fe_face_values.shape_value(i,q_point) *
										fe_face_values.shape_grad(j,q_point)[1] * locally_relevant_solutionV[0](cell->vertex_dof_index(j,0)) *
											fe_face_values.normal_vector(q_point)[0] * fe_face_values.JxW(q_point);
								}//j
						}//q_point
					} else if(dim == 3){
						for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
							Tensor<1,dim> tempX, tempY, tempZ;

							weight = fe_face_values.JxW(q_point);

							for (unsigned int i = 0; i < dofs_per_cell; ++i) {
								iDoFindex = cell->vertex_dof_index(i,0);
								const Tensor<1,dim> Ni_vel_grad = fe_face_values.shape_grad (i, q_point);
								Tensor<1,dim> velocity;
								for(int j = 1; j < dim; ++j)
									velocity[j] = locally_relevant_solutionV[j](iDoFindex);

								tempX[0] += (4.0 / 3.0) * Ni_vel_grad[0] * velocity[0] - (2.0 / 3.0) * Ni_vel_grad[1] * velocity[1]
										-(2.0 / 3.0) * Ni_vel_grad[2] * velocity[2];
								tempX[1] += Ni_vel_grad[1] * velocity[0] + Ni_vel_grad[0] * velocity[1];
								tempX[2] += Ni_vel_grad[2] * velocity[0] + Ni_vel_grad[0] * velocity[2];

								tempY[0] += Ni_vel_grad[1] * velocity[0] + Ni_vel_grad[0] * velocity[1];
								tempY[1] += (-2.0/3.0)*Ni_vel_grad[0] * velocity[0] + (4.0/3.0)*Ni_vel_grad[1] * velocity[1]
										- (2.0/3.0)*Ni_vel_grad[2] * velocity[2];
								tempY[2] += Ni_vel_grad[2] * velocity[1] + Ni_vel_grad[1] * velocity[2];

								tempZ[0] += Ni_vel_grad[2] * velocity[0] + Ni_vel_grad[0] * velocity[2];
								tempZ[1] += Ni_vel_grad[2] * velocity[1] + Ni_vel_grad[1] * velocity[2];
								tempZ[2] += (-2.0/3.0)*Ni_vel_grad[0] * velocity[0] - (2.0/3.0)*Ni_vel_grad[1] * velocity[1]
										+ (4.0/3.0)*Ni_vel_grad[2] * velocity[2];
							}

							for (unsigned int i = 0; i < dofs_per_cell; ++i){
								aux = mu * time_step * fe_face_values.shape_value(i, q_point) * weight;

								local_rhsV[0](i) += aux * tempX * fe_face_values.normal_vector(q_point);
								local_rhsV[1](i) += aux * tempY * fe_face_values.normal_vector(q_point);
								local_rhsV[2](i) += aux * tempZ * fe_face_values.normal_vector(q_point);
							}
						}
					}
				}//if face->at_boundary()

			cell->get_dof_indices (local_dof_indices);
			for(int i = 0; i < dim; ++i)
				constraintsPredV[i].distribute_local_to_global (local_matrixV[i], local_rhsV[i], local_dof_indices, system_mPredV[i], system_rV[i]);
		}//cell

	for(int i = 0; i < dim; ++i){
		system_mPredV[i].compress (VectorOperation::add);
		system_rV[i].compress (VectorOperation::add);
	}
}

template <int dim>
void pfem2Fem<dim>::assemble_pressure_equation()
{
	system_mP = 0.0;
	system_rP = 0.0;

	const double coeff = rho / mainSolver->getParameterHandler().getTimeStep();

	double weight, aux;
	unsigned int jDoFindex;

	for(const auto &cell : dof_handler.active_cell_iterators())
		if(cell->is_locally_owned()){
			fe_values.reinit (cell);
			local_matrixP = 0.0;
			local_rhsP = 0.0;
						
			for (unsigned int q_index = 0; q_index < n_q_points; ++q_index){
				weight = fe_values.JxW (q_index);
				
				for (unsigned int i = 0; i < dofs_per_cell; ++i) {
					const Tensor<1,dim> Nidx_pres = fe_values.shape_grad (i, q_index);

					for (unsigned int j = 0; j < dofs_per_cell; ++j) {
						jDoFindex = cell->vertex_dof_index(j,0);
						
						const Tensor<0,dim> Nj_vel = fe_values.shape_value (j, q_index);
						const Tensor<1,dim> Njdx_pres = fe_values.shape_grad (j, q_index);

						aux = Nidx_pres * Njdx_pres * weight;
						local_matrixP(i,j) += aux;

#ifdef SCHEMEB
						local_rhsP(i) += aux * locally_relevant_old_solutionP(jDoFindex);
#endif
						aux = 0.0;
						for(int k = 0; k < dim; ++k)
							aux += locally_relevant_predictionV[k](jDoFindex) * Nidx_pres[k];

						local_rhsP(i) += coeff * aux * Nj_vel * weight;
					}//j
				}//i
			}//q_index

			for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary() && mainSolver->getPressureDirichletBCpatchIDs().count(cell->face(face_number)->boundary_id()) == 0){
					fe_face_values.reinit (cell, face_number);

					for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point){
						Tensor<1,dim> V_q_point_value;

						for (unsigned int i = 0; i < dofs_per_cell; ++i)
							for(int k = 0; k < dim; ++k)
								V_q_point_value[k] += fe_face_values.shape_value(i, q_point) * locally_relevant_predictionV[k](cell->vertex_dof_index(i,0));
						for (unsigned int i = 0; i < dofs_per_cell; ++i)
							local_rhsP(i) -= coeff * fe_face_values.shape_value(i, q_point) * V_q_point_value *
														fe_face_values.normal_vector(q_point) * fe_face_values.JxW(q_point);
					}
				}

			cell->get_dof_indices (local_dof_indices);
			constraintsP.distribute_local_to_global (local_matrixP, local_rhsP, local_dof_indices, system_mP, system_rP);
		}//cell

	system_mP.compress (VectorOperation::add);
	system_rP.compress (VectorOperation::add);
}

template <int dim>
void pfem2Fem<dim>::assemble_velocity_correction()
{
	for(int i = 0; i < dim; ++i){
		system_mV[i] = 0.0;
		system_rV[i] = 0.0;
	}

	double weight, aux, aux2;
	unsigned int jDoFindex;

	for(const auto &cell : dof_handler.active_cell_iterators())
		if(cell->is_locally_owned()) {
			fe_values.reinit (cell);
			for(int i = 0; i < dim; ++i){
				local_matrixV[i] = 0.0;
				local_rhsV[i] = 0.0;
			}
		
			for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
				weight = fe_values.JxW (q_index);
				
				for (unsigned int i = 0; i < dofs_per_cell; ++i) {
					const Tensor<0,dim> Ni_vel = fe_values.shape_value (i, q_index);
					aux2 = mainSolver->getParameterHandler().getTimeStep() * Ni_vel * weight;
					
					for (unsigned int j = 0; j < dofs_per_cell; ++j) {
						jDoFindex = cell->vertex_dof_index(j,0);
						
						const Tensor<0,dim> Nj_vel = fe_values.shape_value (j, q_index);
						const Tensor<1,dim> Nj_p_grad = fe_values.shape_grad (j, q_index);

						aux = rho * Ni_vel * Nj_vel * weight;
						
						for(int k = 0; k < dim; ++k){
							local_matrixV[k](i,j) += aux;
							local_rhsV[k](i) += aux * locally_relevant_predictionV[k](jDoFindex);

#ifndef SCHEMEB
							local_rhsV[k](i) -= aux2 * Nj_p_grad[k] * locally_relevant_solutionP(jDoFindex);
#else
							local_rhsV[k](i) -= aux2 * Nj_p_grad[k] * (locally_relevant_solutionP(jDoFindex) - locally_relevant_old_solutionP(jDoFindex));
#endif
						}
					}//j
				}//i
			}//q_index
		
			cell->get_dof_indices (local_dof_indices);
			for(int i = 0; i < dim; ++i)
				constraintsV[i].distribute_local_to_global (local_matrixV[i], local_rhsV[i], local_dof_indices, system_mV[i], system_rV[i]);
		}//cell
			
	for(int i = 0; i < dim; ++i){
		system_mV[i].compress (VectorOperation::add);
		system_rV[i].compress (VectorOperation::add);
	}
}

template <int dim>
void pfem2Fem<dim>::fem_step()
{
	TimerOutput::Scope timer_section(mainSolver->getTimer(), "FEM Step");

	for(int i = 0; i < dim; ++i)
		locally_relevant_old_solutionV[i] = locally_relevant_solutionV[i];
	locally_relevant_old_solutionP = locally_relevant_solutionP;

	for(int nOuterCorr = 0; nOuterCorr < outerCorrections; ++nOuterCorr){
		velocity_prediction_bc();
		assemble_velocity_prediction();
		solve_velocity();

		assemble_pressure_equation();
		solve_pressure();

		assemble_velocity_correction();
		solve_velocity(true);
	}
}

template <int dim>
void pfem2Fem<dim>::output_fem_solution(int timestep_number, bool exportPrediction)
{
	DataOut<dim> data_out;

	data_out.attach_dof_handler (dof_handler);
	data_out.add_data_vector (locally_relevant_solutionV[0], "Vx");
	if (dim > 1)
		data_out.add_data_vector (locally_relevant_solutionV[1], "Vy");
	if (dim > 2)
		data_out.add_data_vector (locally_relevant_solutionV[2], "Vz");

	if(exportPrediction){
		data_out.add_data_vector (locally_relevant_predictionV[0], "predVx");
		if (dim > 1)
			data_out.add_data_vector (locally_relevant_predictionV[1], "predVy");
		if (dim > 2)
			data_out.add_data_vector (locally_relevant_predictionV[2], "predVz");
	}

	data_out.add_data_vector (locally_relevant_solutionP, "P");
	
	const auto& tria = mainSolver->getTriangulation();
    Vector<float> subdomain (tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
		subdomain(i) = tria.locally_owned_subdomain();
    data_out.add_data_vector (dof_handler, subdomain, "subdomain");
	
	data_out.build_patches ();

	const std::string filename =  "solution-" + Utilities::int_to_string (timestep_number, 2) + "."
				+ Utilities::int_to_string(Utilities::MPI::this_mpi_process(mainSolver->getCommunicator()), 3) + ".vtu";
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);
    
    if (Utilities::MPI::this_mpi_process(mainSolver->getCommunicator()) == 0) {
        std::vector<std::string> filenames;
        for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mainSolver->getCommunicator()); ++i)
			filenames.push_back ("solution-" + Utilities::int_to_string (timestep_number, 2) + "." + Utilities::int_to_string(i, 3) + ".vtu");

		std::ofstream master_output (("solution-" + Utilities::int_to_string (timestep_number, 2) + ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames);
	}
}

template <int dim>
void pfem2Fem<dim>::initialize_fem_solution()
{
	locally_relevant_solutionP = mainSolver ? mainSolver->getParameterHandler().getPressureInitialValue() : 0.0;
	
	const Tensor<1, dim> initialVelocity = mainSolver ? mainSolver->getParameterHandler().getVelocityInitialValue() : Tensor<1, dim>();
	for (int i = 0; i < dim; ++i)
		locally_relevant_solutionV[i] = initialVelocity[i];
}

template <int dim>
void pfem2Fem<dim>::setPfem2Solver(pfem2Solver<dim> *mainSolver)
{
	const auto& paramHandler = mainSolver->getParameterHandler();

	this->mainSolver = mainSolver;
	this->mu = paramHandler.getDynamicViscosity();
	this->rho = paramHandler.getFluidDensity();
	this->outerCorrections = paramHandler.getOuterIterations();

	if(mainSolver->getNeedLoadsCalculation()){
		this->loadsBoundaryID = paramHandler.getLoadsBoundaryID();
		this->thickness = paramHandler.getThickness();
		this->meanVelocity = paramHandler.getMeanVelocity();
	}

	dof_handler.reinit(mainSolver->getTriangulation());
}

template <int dim>
const DoFHandler<dim> &pfem2Fem<dim>::getDoFhandler() const
{
    return dof_handler;
}

template <int dim>
const std::array<TrilinosWrappers::MPI::Vector, dim> &pfem2Fem<dim>::getSolutionV() const
{
    return locally_relevant_solutionV;
}

template <int dim>
const std::array<TrilinosWrappers::MPI::Vector, dim> &pfem2Fem<dim>::getOldSolutionV() const
{
    return locally_relevant_old_solutionV;
}

template <int dim>
const TrilinosWrappers::MPI::Vector &pfem2Fem<dim>::getSolutionP() const
{
    return locally_relevant_solutionP;
}

template <int dim>
void pfem2Fem<dim>::setSolutionV(const TrilinosWrappers::MPI::Vector &solutionV, unsigned int component)
{
	if(component < dim)
		locally_relevant_solutionV[component] = solutionV;
}

template <int dim>
void pfem2Fem<dim>::setSolutionP(const TrilinosWrappers::MPI::Vector &solutionP)
{
	locally_relevant_solutionP = solutionP;
}

template <int dim>
const unsigned int &pfem2Fem<dim>::getDofsPerCell() const
{
    return dofs_per_cell;
}

template <int dim>
const IndexSet &pfem2Fem<dim>::getLocallyOwnedDofs() const
{
    return locally_owned_dofs;
}

template<int dim>
void pfem2Fem<dim>::solve_velocity(bool correction)
{
	std::array<TrilinosWrappers::MPI::Vector, dim> completely_distributed_solution;
	for(int i = 0; i < dim; ++i)
		completely_distributed_solution[i].reinit(locally_owned_dofs, mainSolver->getCommunicator());
    
    for(int i = 0; i < dim; ++i){
		if(correction) {
			static_cast<TrilinosWrappers::PreconditionJacobi*>(preconditionerV)->initialize (system_mV[i]);
			trilinosSolver->solve (system_mV[i], completely_distributed_solution[i], system_rV[i], *preconditionerV);
		} else {
			static_cast<TrilinosWrappers::PreconditionJacobi*>(preconditionerV)->initialize (system_mPredV[i]);
			trilinosSolver->solve (system_mPredV[i], completely_distributed_solution[i], system_rV[i], *preconditionerV);
		}
		
		if(solver_control.last_check() == SolverControl::success)
			mainSolver->getPcout() << "Solver for V (component " << i << ") converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
		else
			mainSolver->getPcout() << "Solver for V (component " << i << ") failed to converge" << std::endl;
	}
    
    if (correction){
		for(int i = 0; i < dim; ++i){
			constraintsV[i].distribute (completely_distributed_solution[i]);
			locally_relevant_solutionV[i] = completely_distributed_solution[i];
		}
	} else {
		for(int i = 0; i < dim; ++i){
			constraintsPredV[i].distribute (completely_distributed_solution[i]);
			locally_relevant_predictionV[i] = completely_distributed_solution[i];
		}
	}
}

template<int dim>
void pfem2Fem<dim>::solve_pressure()
{
	TrilinosWrappers::MPI::Vector completely_distributed_solution (locally_owned_dofs, mainSolver->getCommunicator());
    
    static_cast<TrilinosWrappers::PreconditionAMG*>(preconditionerP)->initialize (system_mP);
	trilinosSolver->solve (system_mP, completely_distributed_solution, system_rP, *preconditionerP);		
    
    if(solver_control.last_check() == SolverControl::success)
        mainSolver->getPcout() << "Solver for P converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
    else
		mainSolver->getPcout() << "Solver for P failed to converge" << std::endl;
    
	constraintsP.distribute (completely_distributed_solution);
	locally_relevant_solutionP = completely_distributed_solution;
}

template<int dim>
void pfem2Fem<dim>:: calculate_loads(std::ostream &out)
{
	TimerOutput::Scope timer_section(mainSolver->getTimer(), "Loads calculation");

	Tensor<1,dim> F_viscous, F_pressure, C_viscous, C_pressure;
	double point_valueP, dVtdn, weight;

	for(const auto &cell : dof_handler.active_cell_iterators())
		if(cell->is_locally_owned())
			for (unsigned int face_number=0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary() && cell->face(face_number)->boundary_id() == loadsBoundaryID) {
					fe_face_values.reinit (cell, face_number);

					//for 2D only
					for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
						point_valueP = 0.0;
						dVtdn = 0.0;

						weight = fe_face_values.JxW (q_point);

						for (unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_cell; ++vertex){
							point_valueP += locally_relevant_solutionP(cell->vertex_dof_index(vertex,0)) * fe_face_values.shape_value(vertex, q_point);
							dVtdn += (locally_relevant_solutionV[0](cell->vertex_dof_index(vertex,0)) * fe_face_values.normal_vector(q_point)[1] - locally_relevant_solutionV[1](cell->vertex_dof_index(vertex,0)) * fe_face_values.normal_vector(q_point)[0]) *
									(fe_face_values.shape_grad(vertex, q_point)[0] * fe_face_values.normal_vector(q_point)[0] + fe_face_values.shape_grad(vertex, q_point)[1] * fe_face_values.normal_vector(q_point)[1]);
						}//vertex

						F_viscous[0] += mu * dVtdn * fe_face_values.normal_vector(q_point)[1] * weight;
						F_pressure[0] -= point_valueP * fe_face_values.normal_vector(q_point)[0] * weight;
						F_viscous[1] -= mu * dVtdn * fe_face_values.normal_vector(q_point)[0] * weight;
						F_pressure[1] -= point_valueP * fe_face_values.normal_vector(q_point)[1] * weight;
					}//q_index
				}//if

	double coeff = 2.0 / (rho * meanVelocity * meanVelocity * thickness);
	for(int i = 0; i < dim; ++i){
		C_viscous[i] = coeff * F_viscous[i];
		C_pressure[i] = coeff * F_pressure[i];
	}

	double local_coeffs[2 * dim];
	double global_coeffs[2 * dim];
	for(int i = 0; i < dim; ++i){
		local_coeffs[2 * i] = C_viscous[i];
		local_coeffs[2 * i + 1] = C_pressure[i];
	}

	Utilities::MPI::sum(local_coeffs, mainSolver->getCommunicator(), global_coeffs);

	if (Utilities::MPI::this_mpi_process(mainSolver->getCommunicator()) == 0){
		Tensor<1,dim> aerodynamicCoeffs;
		for(int i = 0; i < dim; ++i)
			aerodynamicCoeffs[i] = global_coeffs[2 * i] + global_coeffs[2 * i + 1];

		out << mainSolver->getTime();
		for(int i = 0; i < dim; ++i)
			out << "," << aerodynamicCoeffs[i];

		for(int i = 0; i < dim; ++i)
			out << "," << global_coeffs[2 * i] << "," << global_coeffs[2 * i + 1];

		out << std::endl;
	}
}
