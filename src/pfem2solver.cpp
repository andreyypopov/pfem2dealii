#include "pfem2solver.h"

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

template<int dim>
pfem2Solver<dim>::pfem2Solver(pfem2Fem<dim> *femSolver,
	pfem2ParticleHandler<dim> *particleHandler, pfem2ParameterHandler<dim> *parameterHandler)
	: mpi_communicator (MPI_COMM_WORLD)
	, n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator))
	, pcout (std::cout, (this_mpi_process == 0))
	, tria (mpi_communicator, Triangulation<dim>::maximum_smoothing)
	, timer (mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
{
	this->femSolver = femSolver;
	this->particleHandler = particleHandler;
	this->parameterHandler = parameterHandler;
	
	parameterHandler->readParameters("input_data.prm");
	
	time = 0.0;
	timestep_number = 0;
    time_step = parameterHandler->getTimeStep();
	final_time = parameterHandler->getFinalTime();
	needLoadsCalculation = parameterHandler->getLoadsBoundaryID() != 0;

	femSolver->setPfem2Solver(this);
	particleHandler->setPfem2Solver(this);
}

template<int dim>
pfem2Solver<dim>::~pfem2Solver()
{
}

template<int dim>
const MPI_Comm pfem2Solver<dim>::getCommunicator() const
{
	return mpi_communicator;
}

template<int dim>
const parallel::distributed::Triangulation<dim>& pfem2Solver<dim>::getTriangulation() const
{
	return tria;
}

template <int dim>
const pfem2Fem<dim> &pfem2Solver<dim>::getFemSolver() const
{
    return *femSolver;
}

template <int dim>
const pfem2ParameterHandler<dim> &pfem2Solver<dim>::getParameterHandler() const
{
    return *parameterHandler;
}

template <int dim>
pfem2Fem<dim> &pfem2Solver<dim>::getFemSolver()
{
    return *femSolver;
}

template <int dim>
const ConditionalOStream &pfem2Solver<dim>::getPcout() const
{
    return pcout;
}

template <int dim>
TimerOutput &pfem2Solver<dim>::getTimer()
{
    return timer;
}

template <int dim>
const std::set<unsigned int> &pfem2Solver<dim>::getVelocityDirichletBCpatchIDs() const
{
    return velocityDirichletBCpatchIDs;
}

template <int dim>
const std::set<unsigned int> &pfem2Solver<dim>::getPressureDirichletBCpatchIDs() const
{
    return pressureDirichletBCpatchIDs;
}

template <int dim>
const int &pfem2Solver<dim>::getTimestepNumber() const
{
    return timestep_number;
}

template <int dim>
const bool &pfem2Solver<dim>::getNeedLoadsCalculation() const
{
    return needLoadsCalculation;
}

template<int dim>
void pfem2Solver<dim>::build_mesh(bool outputAfterBuild)
{
	TimerOutput::Scope timer_section(timer, "Mesh import");
    
    //TO DO: check whether file exists
    
    GridIn<dim> gridin;
    gridin.attach_triangulation(tria);
    std::ifstream f(parameterHandler->getMeshFileName());
    gridin.read_unv(f);
    
    pcout << "Imported grid contains " << tria.n_cells() << " cells" << std::endl;
    
    if (outputAfterBuild){
		GridOut grid_out;

		std::ofstream out ("Mesh.eps");
		grid_out.write_eps (tria, out);
		std::cout << "Grid written to EPS" << std::endl;
		
		std::ofstream out2 ("Mesh.vtk");
		grid_out.write_vtk (tria, out2);
		std::cout << "Grid written to VTK" << std::endl;
	}
}

template<int dim>
void pfem2Solver<dim>::run()
{
    if (this_mpi_process == 0){
		system("rm solution*vtu");
		system("rm particles*vtu");

		if(needLoadsCalculation){
			forcesFile.open("forces.csv");
			if(dim == 2)
				forcesFile << "t,Cx,Cy,Cx_nu,Cx_p,Cy_nu,Cy_p" << std::endl;
		}
	}

	build_mesh();
	femSolver->setup_system();
	femSolver->initialize_fem_solution();

	particleHandler->seed_particles();
	particleHandler->initialize_maps();

	output_results(parameterHandler->getOutputParticles());

	for (; time <= final_time; time += time_step) {
        pcout << std::endl << "Time step no. " << ++timestep_number << " at t = " << time << std::endl;
        
		particleHandler->correct_particle_velocity();
		particleHandler->move_particles();
		particleHandler->project_particle_fields();
    
		femSolver->fem_step();
        if(timestep_number % parameterHandler->getResultsOutputFrequency() == 0)
			output_results(parameterHandler->getOutputParticles());

		if(needLoadsCalculation)
			femSolver->calculate_loads(forcesFile);

        timer.print_summary();
    }//time

    if(this_mpi_process == 0 && needLoadsCalculation){
		forcesFile.flush();
		forcesFile.close();
	}
}

template<int dim>
void pfem2Solver<dim>::output_results(bool exportParticles, bool exportPrediction)
{
	TimerOutput::Scope timer_section(timer, "Results output");
    
	femSolver->output_fem_solution(timestep_number, exportPrediction);
	if(exportParticles)
		particleHandler->output_particle_solution(timestep_number);
}
