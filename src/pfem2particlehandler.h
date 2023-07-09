#ifndef PFEM2PARTICLEHANDLER_H
#define PFEM2PARTICLEHANDLER_H

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include "pfem2particle.h"

using namespace dealii;

template<int dim>
using pfem2ParticleIterator = typename std::vector<pfem2Particle<dim>>::iterator;

template<int dim>
class pfem2Solver;

template<int dim>
class pfem2ParticleHandler
{
public:
	pfem2ParticleHandler(const FE_Q<dim> *finite_element);
	virtual ~pfem2ParticleHandler();
	
	void clear_particles();
	
	pfem2ParticleIterator<dim> remove_particle(pfem2ParticleIterator<dim> particleIndex);
	void insert_particle(pfem2Particle<dim> &particle, const typename DoFHandler<dim>::active_cell_iterator &cell);
		
    pfem2ParticleIterator<dim> begin();
    pfem2ParticleIterator<dim> end();
    
    void initialize_maps();
    
    const pfem2Solver<dim>* getPfem2Solver() const;
	void setPfem2Solver(pfem2Solver<dim> *mainSolver);

    virtual void seed_particles();
    virtual void correct_particle_velocity();
    void move_particles();
    void project_particle_fields();

    void output_particle_solution(int timestep_number);

protected:
    unsigned int fill_cell_parts_indices();

    std::vector<std::array<unsigned int, dim>> cellPartsIndices;

    std::vector<pfem2Particle<dim>> particles;

    std::array<unsigned int, dim> quantities;
	unsigned int maxParticlesPerCellPart;
    int particleCount;

    unsigned int particle_integration_steps;
    double particle_transport_timestep;

	pfem2Solver<dim>* mainSolver;

    const FE_Q<dim> *feq;

private:
    void sort_particles_into_subdomains_and_cells(const DoFHandler<dim> &dof_handler);
    void check_particle_distribution(const DoFHandler<dim> &dof_handler);

#ifdef DEAL_II_WITH_MPI
	void send_recv_particles(const std::map<unsigned int, std::vector<pfem2Particle<dim>>> &particles_to_send);
#endif

    MappingQ1<dim> mapping;

    std::vector<std::set<typename Triangulation<dim>::active_cell_iterator>> vertex_to_cells;
    std::vector<std::vector<Tensor<1,dim>>> vertex_to_cell_centers;
};

template<int dim>
bool compare_particle_association(const unsigned int a, const unsigned int b, const Tensor<1,dim> &particle_direction, const std::vector<Tensor<1,dim>> &center_directions);

template class pfem2ParticleHandler<2>;

template class pfem2ParticleHandler<3>;

#endif // PFEM2PARTICLEHANDLER_H
