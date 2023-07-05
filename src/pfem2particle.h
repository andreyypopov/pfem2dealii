#ifndef PFEM2PARTICLE_H
#define PFEM2PARTICLE_H

#include <deal.II/base/tensor.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q1.h>

using namespace dealii;

template<int dim>
class pfem2Particle
{
public:
	pfem2Particle();
	pfem2Particle(const Point<dim> &location,const Point<dim> &reference_location, const unsigned int id);
	pfem2Particle(const void* &begin_data);
	
	void set_location (const Point<dim> &new_location);
	const Point<dim> &get_location () const;

	void set_reference_location (const Point<dim> &new_reference_location);
	const Point<dim> &get_reference_location () const;
	
	unsigned int get_id () const;
	
	void set_tria_position (const int &new_position);
	int get_tria_position () const;
		
	void set_velocity (const Tensor<1,dim> &new_velocity);
	void set_velocity_component (const double value, int component);
	
	const Tensor<1,dim> &get_velocity() const;
	double get_velocity_component(int component) const;
	
	const Tensor<1,dim> &get_velocity_ext() const;
	void set_velocity_ext (const Tensor<1,dim> &new_ext_velocity);

	int get_cell_dof(const unsigned int vertex_index) const;
	void set_cell_dofs(const typename DoFHandler<dim>::active_cell_iterator &cell);

	typename Triangulation<dim>::cell_iterator get_surrounding_cell(const Triangulation<dim> &triangulation) const;
	typename DoFHandler<dim>::cell_iterator get_surrounding_cell(const Triangulation<dim> &triangulation, const DoFHandler<dim> &dof_handler) const;
	
	unsigned int find_closest_vertex_of_cell(const typename Triangulation<dim>::active_cell_iterator &cell, const Mapping<dim> &mapping);
	
	std::size_t serialized_size_in_bytes() const;
	
	void write_data(void* &data) const;

private:
	Point<dim> location;
	Point<dim> reference_location;
	unsigned int id;

	int cell_dofs[GeometryInfo<dim>::vertices_per_cell];

	int tria_position;

	Tensor<1,dim> velocity;						 //!< Velocity transferred by the particle
	Tensor<1,dim> velocity_ext;					 //!< Velocity for particle transport (external velocity)
};

template class pfem2Particle<2>;

template class pfem2Particle<3>;

#endif // PFEM2PARTICLE_H
