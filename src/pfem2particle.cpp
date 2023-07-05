#include "pfem2particle.h"

#include <deal.II/particles/property_pool.h>

using namespace dealii;

template<int dim>
pfem2Particle<dim>::pfem2Particle(const Point<dim> &location, const Point<dim> &reference_location, const unsigned id)
	: id (id)
{
	this->location = location;
	this->reference_location = reference_location;
	this->velocity = 0;
	this->velocity_ext = 0;
}

template<int dim>
pfem2Particle<dim>::pfem2Particle(const void* &data)
{
	const unsigned int *id_data = static_cast<const unsigned int*> (data);
    id = *id_data++;
    
    const int *triaData = reinterpret_cast<const int*> (id_data);
    tria_position = *triaData++;
    
    for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
		cell_dofs[i] = *triaData++;

    const double *pdata = reinterpret_cast<const double*> (triaData);

    for (unsigned int i = 0; i < dim; ++i) 
		location[i] = *pdata++;

    for (unsigned int i = 0; i < dim; ++i)
		reference_location[i] = *pdata++;

    for (unsigned int i = 0; i < dim; ++i)
		velocity[i] = *pdata++;
    
	for (unsigned int i = 0; i < dim; ++i)
		velocity_ext[i] = *pdata++;

    data = static_cast<const void*> (pdata);
}

template<int dim>
pfem2Particle<dim>::pfem2Particle()
	: pfem2Particle(Point<dim>(), Point<dim>(), 0)
{
}

template<int dim>
void pfem2Particle<dim>::set_location (const Point<dim> &new_location)
{
	this->location = new_location;
}

template<int dim>
const Point<dim> &pfem2Particle<dim>::get_location () const
{
    return location;
}

template<int dim>
void pfem2Particle<dim>::set_reference_location (const Point<dim> &new_reference_location)
{
    this->reference_location = new_reference_location;
}

template<int dim>
const Point<dim> &pfem2Particle<dim>::get_reference_location () const
{
	return reference_location;
}

template<int dim>
unsigned int pfem2Particle<dim>::get_id () const
{
    return id;
}

template<int dim>
int pfem2Particle<dim>::get_cell_dof(const unsigned int index) const
{
	return (index < GeometryInfo<dim>::vertices_per_cell) ? cell_dofs[index] : -1;
}

template<int dim>
void pfem2Particle<dim>::set_cell_dofs(const typename DoFHandler<dim>::active_cell_iterator &cell)
{
	for(unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
		cell_dofs[i] = cell->vertex_dof_index(i, 0);
}

template<int dim>
int pfem2Particle<dim>::get_tria_position() const
{
	return tria_position;
}

template<int dim>
void pfem2Particle<dim>::set_tria_position(const int &new_position)
{
	tria_position = new_position;
}

template<int dim>
const Tensor<1,dim> &pfem2Particle<dim>::get_velocity() const
{
	return velocity;
}

template<int dim>
const Tensor<1,dim> &pfem2Particle<dim>::get_velocity_ext() const
{
	return velocity_ext;
}

template<int dim>
double pfem2Particle<dim>::get_velocity_component(int component) const
{
	return velocity[component];
}

template<int dim>
void pfem2Particle<dim>::set_velocity (const Tensor<1,dim> &new_velocity)
{
	velocity = new_velocity;
}

template<int dim>
void pfem2Particle<dim>::set_velocity_component (const double value, int component)
{
	velocity[component] = value;
}

template<int dim>
void pfem2Particle<dim>::set_velocity_ext (const Tensor<1,dim> &new_ext_velocity)
{
	velocity_ext = new_ext_velocity;
}

template<int dim>
typename Triangulation<dim>::cell_iterator pfem2Particle<dim>::get_surrounding_cell(const Triangulation<dim> &triangulation) const
{
	const typename Triangulation<dim>::cell_iterator cell(&triangulation, triangulation.n_levels() - 1, tria_position);
	
	return cell;
}

template<int dim>
typename DoFHandler<dim>::cell_iterator pfem2Particle<dim>::get_surrounding_cell(const Triangulation<dim> &triangulation, const DoFHandler<dim> &dof_handler) const
{
	const typename DoFHandler<dim>::cell_iterator cell(&triangulation, triangulation.n_levels() - 1, tria_position, &dof_handler);
	
	return cell;
}


template<int dim>
unsigned int pfem2Particle<dim>::find_closest_vertex_of_cell(const typename Triangulation<dim>::active_cell_iterator &cell, const Mapping<dim> &mapping)
{
	//transformation of local particle coordinates transformation is required as the global particle coordinates have already been updated by the time this function is called
	const Point<dim> old_position = mapping.transform_unit_to_real_cell(cell, reference_location);
	
	Tensor<1,dim> velocity_normalized = velocity_ext / velocity_ext.norm();
	Tensor<1,dim> particle_to_vertex = cell->vertex(0) - old_position;
    particle_to_vertex /= particle_to_vertex.norm();
    
    double maximum_angle = velocity_normalized * particle_to_vertex;
    unsigned int closest_vertex = 0;
    
    for (unsigned int v = 1; v < GeometryInfo<dim>::vertices_per_cell; ++v){
		particle_to_vertex = cell->vertex(v) - old_position;
		particle_to_vertex /= particle_to_vertex.norm();
		const double v_angle = velocity_normalized * particle_to_vertex;
		
		if (v_angle > maximum_angle){
			closest_vertex = v;
			maximum_angle = v_angle;
		}
	}
	
	return closest_vertex;
}

template<int dim>
std::size_t pfem2Particle<dim>::serialized_size_in_bytes() const
{
	std::size_t size = sizeof(id) + sizeof(location) + sizeof(reference_location) + sizeof(tria_position)
		+ sizeof(cell_dofs) + sizeof(velocity) + sizeof(velocity_ext);

	return size;
}

template<int dim>
void pfem2Particle<dim>::write_data (void* &data) const
{
	unsigned int *id_data  = static_cast<unsigned int*> (data);
    *id_data = id;
    ++id_data;
    
    int *triaData = reinterpret_cast<int*> (id_data);
    *triaData = tria_position;
    ++triaData;
    
    // Write cell DoFs numbers
    for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i,++triaData)
		*triaData = cell_dofs[i];
    
    double *pdata = reinterpret_cast<double*> (triaData);

    // Write location data
    for (unsigned int i = 0; i < dim; ++i,++pdata)
		*pdata = location[i];

    // Write reference location data
    for (unsigned int i = 0; i < dim; ++i,++pdata)
		*pdata = reference_location[i];
    
    // Write velocity
    for (unsigned int i = 0; i < dim; ++i,++pdata)
		*pdata = velocity[i];
    
    // Write streamline velocity
    for (unsigned int i = 0; i < dim; ++i,++pdata)
		*pdata = velocity_ext[i];
     
    data = static_cast<void*> (pdata);
}

